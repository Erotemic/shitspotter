package io.kitware.shitspotter.desktop

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureType
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.StubDetectorBackend
import io.kitware.shitspotter.ui.AppRootTheme
import io.kitware.shitspotter.ui.AppScreen
import kotlinx.datetime.Clock
import java.io.File

private fun loadStillImageFromArgs(args: Array<String>): File? {
    args.forEach { a ->
        if (a.startsWith("--image=")) return File(a.removePrefix("--image="))
    }
    System.getenv("SHITSPOTTER_DESKTOP_IMAGE")?.let { return File(it) }
    System.getProperty("desktopHarnessImage")?.let { return File(it) }
    return null
}

fun main(args: Array<String>) {
    PrintlnLogger.info("ShitSpotter.Desktop", "starting; commit=${BuildInfo.appCommit}")

    val state = AppState()
    val backend: DetectorBackend = StubDetectorBackend()
    val imageFile = loadStillImageFromArgs(args)
    val frame = imageFile?.takeIf { it.isFile }?.let { StillImageFrameSource.fromFile(it) }
    val backgroundImage = imageFile?.takeIf { it.isFile }?.let { javax.imageio.ImageIO.read(it) }
    val failureStore = DesktopFailureCaseStore(File("failure_cases"))

    val harness = DesktopHarness(state) { backend }

    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "ShitSpotter (desktop harness)",
        ) {
            val scope = rememberCoroutineScope()
            LaunchedEffect(Unit) {
                if (frame != null) {
                    PrintlnLogger.info("ShitSpotter.Desktop", "looping over still image ${frame.width}x${frame.height}")
                    harness.runLoop(scope, frame)
                } else {
                    PrintlnLogger.warn(
                        "ShitSpotter.Desktop",
                        "no input image; pass --image=<path> or set SHITSPOTTER_DESKTOP_IMAGE",
                    )
                }
            }
            AppRootTheme {
                AppScreen(
                    state = state,
                    cameraSurface = remember { DesktopCameraSurface(backgroundImage) },
                    onSaveFailureCase = { type, note ->
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
                        val path = failureStore.save(ByteArray(0), md)
                        state.failureCasesSavedCount = state.failureCasesSavedCount + 1
                        PrintlnLogger.info("ShitSpotter.Desktop", "failure-case saved → $path")
                    },
                )
            }
        }
    }
}
