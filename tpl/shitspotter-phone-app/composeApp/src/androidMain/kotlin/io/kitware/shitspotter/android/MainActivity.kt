package io.kitware.shitspotter.android

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.LocationManager
import android.media.MediaActionSound
import android.net.Uri
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.CaptureLabel
import io.kitware.shitspotter.core.CaptureMetadata
import io.kitware.shitspotter.core.CaptureReviewEntry
import io.kitware.shitspotter.core.MetadataMode
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.SettingsStore
import io.kitware.shitspotter.core.applySettings
import io.kitware.shitspotter.core.toSettings
import io.kitware.shitspotter.ui.AppRootTheme
import io.kitware.shitspotter.ui.AppScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock
import java.io.File

class MainActivity : ComponentActivity() {

    private val state = AppState()
    private lateinit var settingsStore: SettingsStore
    private lateinit var backendManager: AndroidBackendManager
    private lateinit var photoStore: PhotoStore
    private val captureExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private val shutterSound = MediaActionSound()
    private var torchOn = androidx.compose.runtime.mutableStateOf(false)
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
    private val cameraPermissionGranted = androidx.compose.runtime.mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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

                    val torchState = torchOn
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
                        onTakePhoto = { label, note -> savePhoto(label, note) },
                        onToggleTorch = {
                            torchState.value = !torchState.value
                            androidSurface?.setTorch(torchState.value)
                        },
                        torchOn = torchState.value,
                        onReviewPhotos = ::openReview,
                        onUpdatePhotoLabel = ::updatePhotoLabel,
                        onSharePhoto = ::sharePhoto,
                        onShareAllPhotos = ::shareAllPhotos,
                        onSharePhotos = ::sharePhotos,
                        onDeletePhoto = ::deletePhoto,
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
        shutterSound.play(MediaActionSound.SHUTTER_CLICK)
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

    private fun openReview() {
        lifecycleScope.launch {
            val entries = withContext(Dispatchers.IO) {
                photoStore.listAll().map { (file, meta) ->
                    CaptureReviewEntry(
                        filePath = file.absolutePath,
                        timestamp = meta?.timestamp ?: file.nameWithoutExtension,
                        label = meta?.label ?: CaptureLabel.UNCERTAIN,
                        detectionCount = meta?.detections?.size ?: 0,
                        note = meta?.userNote,
                    )
                }
            }
            state.capturedPhotos = entries
            state.reviewMode = true
        }
    }

    private fun sharePhoto(filePath: String) {
        val jpegFile = File(filePath)
        val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
        val files = buildList {
            add(jpegFile)
            if (jsonFile.exists()) add(jsonFile)
        }
        sendFilesViaEmail(files, "ShitSpotter — 1 photo")
    }

    private fun sharePhotos(filePaths: List<String>) {
        val files = buildList {
            for (path in filePaths) {
                val jpegFile = File(path)
                add(jpegFile)
                val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
                if (jsonFile.exists()) add(jsonFile)
            }
        }
        if (files.isEmpty()) { state.setError("No files to send"); return }
        sendFilesViaEmail(files, "ShitSpotter — ${filePaths.size} photo(s)")
    }

    private fun shareAllPhotos() {
        lifecycleScope.launch {
            val entries = withContext(Dispatchers.IO) { photoStore.listAll() }
            if (entries.isEmpty()) {
                state.setError("No photos to send")
                return@launch
            }
            val files = buildList {
                for ((jpegFile, _) in entries) {
                    add(jpegFile)
                    val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
                    if (jsonFile.exists()) add(jsonFile)
                }
            }
            sendFilesViaEmail(files, "ShitSpotter — ${entries.size} photo(s)")
        }
    }

    private fun sendFilesViaEmail(files: List<File>, subject: String) {
        val uris = ArrayList<Uri>()
        for (file in files) {
            try {
                uris.add(FileProvider.getUriForFile(this, "$packageName.fileprovider", file))
            } catch (e: Throwable) {
                PrintlnLogger.warn("MainActivity", "FileProvider skipped ${file.name}: ${e.message}")
            }
        }
        if (uris.isEmpty()) {
            state.setError("Could not prepare files for sharing")
            return
        }
        val recipient = state.recipientEmail.trim()
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "*/*"
            if (recipient.isNotEmpty()) putExtra(Intent.EXTRA_EMAIL, arrayOf(recipient))
            putExtra(Intent.EXTRA_SUBJECT, subject)
            putExtra(
                Intent.EXTRA_TEXT,
                "ShitSpotter capture data. See attached JPEG + JSON metadata.",
            )
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivity(Intent.createChooser(intent, "Send via email"))
        } catch (_: ActivityNotFoundException) {
            state.setError("No app found to handle email sharing")
        }
    }

    private fun deletePhoto(filePath: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            photoStore.delete(File(filePath))
            val entries = photoStore.listAll().map { (file, meta) ->
                CaptureReviewEntry(
                    filePath = file.absolutePath,
                    timestamp = meta?.timestamp ?: file.nameWithoutExtension,
                    label = meta?.label ?: CaptureLabel.UNCERTAIN,
                    detectionCount = meta?.detections?.size ?: 0,
                    note = meta?.userNote,
                )
            }
            withContext(Dispatchers.Main) {
                state.capturedPhotos = entries
            }
        }
    }

    private fun updatePhotoLabel(filePath: String, label: CaptureLabel, note: String?) {
        lifecycleScope.launch(Dispatchers.IO) {
            photoStore.updateLabel(File(filePath), label, note)
            val entries = photoStore.listAll().map { (file, meta) ->
                CaptureReviewEntry(
                    filePath = file.absolutePath,
                    timestamp = meta?.timestamp ?: file.nameWithoutExtension,
                    label = meta?.label ?: CaptureLabel.UNCERTAIN,
                    detectionCount = meta?.detections?.size ?: 0,
                    note = meta?.userNote,
                )
            }
            withContext(Dispatchers.Main) {
                state.capturedPhotos = entries
            }
        }
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
        try { shutterSound.release() } catch (_: Throwable) {}
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
                "The app does not record or upload video. Captures are kept on the device.",
                color = Color(0xFFCCCCCC),
            )
            Button(onClick = onRequest) { Text("Grant camera permission") }
        }
    }
}
