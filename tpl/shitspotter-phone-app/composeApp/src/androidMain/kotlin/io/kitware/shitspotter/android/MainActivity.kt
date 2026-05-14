package io.kitware.shitspotter.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.CaptureLabel
import io.kitware.shitspotter.core.CaptureMetadata
import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.MetadataMode
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.SettingsStore
import io.kitware.shitspotter.core.applySettings
import io.kitware.shitspotter.core.toSettings
import io.kitware.shitspotter.ui.AppRootTheme
import io.kitware.shitspotter.ui.AppScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock
import java.io.File

class MainActivity : ComponentActivity() {

    private val state = AppState()
    private lateinit var failureStore: AndroidFailureCaseStore
    private lateinit var settingsStore: SettingsStore
    private lateinit var backendManager: AndroidBackendManager
    private lateinit var photoStore: PhotoStore
    private val captureExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private var paused by mutableStateOf(false)
    private var torchOn by mutableStateOf(false)
    private var androidSurface: AndroidCameraSurface? = null
    private var lastLocation: android.location.Location? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermissionGranted.value = granted
        if (granted) requestLocationPermission()
    }
    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { _ -> }
    private val cameraPermissionGranted = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        failureStore = AndroidFailureCaseStore(this)
        settingsStore = AndroidSettingsStore(this)
        state.applySettings(settingsStore.load())
        backendManager = AndroidBackendManager(this)
        backendManager.setActive(state.activeModelId)
        photoStore = PhotoStore(File(getExternalFilesDir("pictures") ?: filesDir, "shitspotter"))

        val have = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        cameraPermissionGranted.value = have
        if (!have) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            requestLocationPermission()
        }

        setContent {
            AppRootTheme {
                if (!cameraPermissionGranted.value) {
                    PermissionRationale(onRequest = {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    })
                } else {
                    LaunchedEffect(Unit) {
                        snapshotFlow { state.activeModelId }
                            .distinctUntilChanged()
                            .collect { id ->
                                withContext(Dispatchers.IO) {
                                    backendManager.setActive(id)
                                }
                            }
                    }

                    val surface = remember {
                        AndroidCameraSurface(
                            context = this@MainActivity,
                            lifecycleOwner = this@MainActivity,
                            state = state,
                            backendManager = backendManager,
                        ).also { androidSurface = it }
                    }
                    AppScreen(
                        state = state,
                        cameraSurface = surface,
                        onSaveFailureCase = ::saveFailureCase,
                        onTogglePause = {
                            paused = !paused
                            androidSurface?.setPaused(paused)
                        },
                        isPaused = paused,
                        canSaveFailureCase = androidSurface?.hasAnalyzedFrame() ?: false,
                        onTakePhoto = { label, note -> savePhoto(label, note) },
                        onToggleTorch = {
                            torchOn = !torchOn
                            androidSurface?.setTorch(torchOn)
                        },
                        torchOn = torchOn,
                    )
                }
            }
        }
    }

    override fun onStop() {
        super.onStop()
        androidSurface?.setPaused(true)
    }

    override fun onStart() {
        super.onStart()
        androidSurface?.setPaused(false)
    }

    private fun requestLocationPermission() {
        val fineGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        val coarseGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        if (!fineGranted || !coarseGranted) {
            locationPermissionLauncher.launch(
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                ),
            )
        }
    }

    private fun getLastLocation(): android.location.Location? {
        val hasFine = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        val hasCoarse = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        if (!hasFine && !hasCoarse) return null
        val lm = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val providers = lm.getProviders(true)
        var best: android.location.Location? = null
        for (provider in providers) {
            try {
                val loc = lm.getLastKnownLocation(provider) ?: continue
                if (best == null || loc.accuracy < best.accuracy) best = loc
            } catch (_: SecurityException) {}
        }
        lastLocation = best ?: lastLocation
        return lastLocation
    }

    private fun savePhoto(label: CaptureLabel, note: String?) {
        val surface = androidSurface ?: return
        val now = Clock.System.now().toString()
        val outputFile = photoStore.newCaptureFile(now)
        val backendSnap = backendManager.snapshot()
        val loc = if (state.metadataMode == MetadataMode.FULL) getLastLocation() else null
        surface.takePicture(
            outputFile = outputFile,
            executor = captureExecutor,
            onSuccess = { file ->
                val metadata = CaptureMetadata(
                    timestamp = now,
                    label = label,
                    modelId = backendSnap.spec.modelId,
                    modelHash = backendSnap.spec.modelHash,
                    appCommit = BuildInfo.appCommit,
                    buildDate = BuildInfo.buildDate,
                    deviceModel = BuildInfo.deviceModel,
                    scoreThreshold = state.scoreThreshold,
                    detections = state.lastDetections,
                    latitude = loc?.latitude,
                    longitude = loc?.longitude,
                    userNote = note,
                )
                photoStore.addMetadataAndSave(file, metadata, state.metadataMode, loc)
                state.photosSavedCount++
                PrintlnLogger.info("MainActivity", "photo saved → $file")
            },
            onError = { e ->
                state.setError("Photo capture failed: ${e.message}")
            },
        )
    }

    private fun saveFailureCase(type: FailureType, note: String?) {
        val surface = androidSurface
        val backendSnap = backendManager.snapshot()
        if (surface == null || !surface.hasAnalyzedFrame()) {
            PrintlnLogger.warn(
                "ShitSpotter.Failure",
                "save-failure tapped before first frame; nothing to write",
            )
            state.setError("no frame yet; wait for the camera to start")
            return
        }
        val tele = state.lastTelemetry
        val now = Clock.System.now().toString()
        val md = FailureCaseMetadata(
            timestamp = now,
            deviceModel = BuildInfo.deviceModel,
            osVersion = BuildInfo.osVersion,
            appCommit = BuildInfo.appCommit,
            modelId = backendSnap.spec.modelId,
            modelHash = backendSnap.spec.modelHash,
            runtimeBackend = backendSnap.backendName,
            delegate = backendSnap.delegate,
            inputWidth = backendSnap.spec.inputWidth,
            inputHeight = backendSnap.spec.inputHeight,
            scoreThreshold = state.scoreThreshold,
            iouThreshold = backendSnap.spec.iouThreshold,
            latencyMs = tele?.totalMs ?: 0.0,
            fpsRecent = tele?.fpsRecent ?: 0.0,
            failureType = type,
            userNote = note,
            detections = state.lastDetections,
        )
        val jpegBytes = surface.encodeLastFrameAsJpeg()
        if (jpegBytes.isEmpty()) {
            PrintlnLogger.warn(
                "ShitSpotter.Failure",
                "encodeLastFrameAsJpeg returned 0 bytes; skipping save",
            )
            state.setError("frame encode failed; not saving")
            return
        }
        val path = failureStore.save(jpegBytes, md)
        state.failureCasesSavedCount = state.failureCasesSavedCount + 1
        PrintlnLogger.info("ShitSpotter.Failure", "saved → $path")
    }

    override fun onPause() {
        super.onPause()
        try { settingsStore.save(state.toSettings()) } catch (_: Throwable) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        try { captureExecutor.shutdown() } catch (_: Throwable) {}
        try { androidSurface?.close() } catch (_: Throwable) {}
        try { backendManager.close() } catch (_: Throwable) {}
    }
}

@Composable
private fun PermissionRationale(onRequest: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101010)),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(24.dp),
        ) {
            Text(
                "ShitSpotter needs camera access to run live, fully-on-device detection.",
                color = Color.White,
            )
            Text(
                "The app does not record or upload video. Failure-case captures are kept on the device.",
                color = Color(0xFFCCCCCC),
            )
            Button(onClick = onRequest) { Text("Grant camera permission") }
        }
    }
}
