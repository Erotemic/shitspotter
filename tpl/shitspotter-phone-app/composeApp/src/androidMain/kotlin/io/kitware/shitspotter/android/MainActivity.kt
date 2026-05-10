package io.kitware.shitspotter.android

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.SettingsStore
import io.kitware.shitspotter.core.StubDetectorBackend
import io.kitware.shitspotter.core.applySettings
import io.kitware.shitspotter.core.toSettings
import io.kitware.shitspotter.ui.AppRootTheme
import io.kitware.shitspotter.ui.AppScreen
import kotlinx.datetime.Clock

class MainActivity : ComponentActivity() {

    private val state = AppState()
    private lateinit var failureStore: AndroidFailureCaseStore
    private lateinit var settingsStore: SettingsStore
    private var backend: DetectorBackend = StubDetectorBackend()
    private var paused by mutableStateOf(false)
    private var androidSurface: AndroidCameraSurface? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        cameraPermissionGranted.value = granted
    }
    private val cameraPermissionGranted = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        failureStore = AndroidFailureCaseStore(this)
        settingsStore = AndroidSettingsStore(this)
        state.applySettings(settingsStore.load())
        backend = chooseBackend(this)

        val have = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        cameraPermissionGranted.value = have
        if (!have) cameraPermissionLauncher.launch(Manifest.permission.CAMERA)

        setContent {
            AppRootTheme {
                if (!cameraPermissionGranted.value) {
                    PermissionRationale(onRequest = {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    })
                } else {
                    val surface = remember {
                        AndroidCameraSurface(
                            context = this@MainActivity,
                            lifecycleOwner = this@MainActivity,
                            state = state,
                            backendProvider = { backend },
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
                    )
                }
            }
        }
    }

    private fun saveFailureCase(type: FailureType, note: String?) {
        val tele = state.lastTelemetry
        val now = Clock.System.now().toString()
        val md = FailureCaseMetadata(
            timestamp = now,
            deviceModel = BuildInfo.deviceModel,
            osVersion = BuildInfo.osVersion,
            appCommit = BuildInfo.appCommit,
            modelId = backend.spec.modelId,
            modelHash = backend.spec.modelHash,
            runtimeBackend = backend.backendName,
            delegate = backend.delegate,
            inputWidth = backend.spec.inputWidth,
            inputHeight = backend.spec.inputHeight,
            scoreThreshold = backend.spec.scoreThreshold,
            iouThreshold = backend.spec.iouThreshold,
            latencyMs = tele?.totalMs ?: 0.0,
            fpsRecent = tele?.fpsRecent ?: 0.0,
            failureType = type,
            userNote = note,
            detections = state.lastDetections,
        )
        val jpegBytes = androidSurface?.encodeLastFrameAsJpeg() ?: ByteArray(0)
        val path = failureStore.save(jpegBytes, md)
        state.failureCasesSavedCount = state.failureCasesSavedCount + 1
        PrintlnLogger.info("ShitSpotter.Failure", "saved → $path")
    }

    override fun onDestroy() {
        super.onDestroy()
        try { backend.close() } catch (_: Throwable) {}
    }

    override fun onPause() {
        super.onPause()
        try { settingsStore.save(state.toSettings()) } catch (_: Throwable) {}
    }

    private fun chooseBackend(ctx: android.content.Context): DetectorBackend {
        val candidate: ModelSpec = ModelSpec.YOLOX_NANO_POOP
        val loader = AndroidModelLoader(ctx)
        val file = loader.resolveOrCopy(candidate) ?: run {
            PrintlnLogger.warn(
                "ShitSpotter.MainActivity",
                "no ${candidate.modelFile} found in external/cache/assets — using stub detector",
            )
            return StubDetectorBackend()
        }
        return try {
            val hash = try { loader.sha256(file).take(16) } catch (_: Throwable) { null }
            val resolved = candidate.copy(modelHash = hash)
            OnnxRuntimeAndroidBackend(resolved, file.absolutePath, tryNnapi = true)
                .also { it.warmup() }
        } catch (t: Throwable) {
            PrintlnLogger.error("ShitSpotter.MainActivity", "ONNX init failed; falling back to stub", t)
            StubDetectorBackend()
        }
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
