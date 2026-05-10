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
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.StubDetectorBackend
import io.kitware.shitspotter.core.applySettings
import io.kitware.shitspotter.core.toSettings
import io.kitware.shitspotter.ui.AppRootTheme
import io.kitware.shitspotter.ui.AppScreen
import kotlinx.datetime.Clock
import java.io.File

private fun argValue(args: Array<String>, key: String): String? {
    val prefix = "$key="
    args.forEach { if (it.startsWith(prefix)) return it.removePrefix(prefix) }
    return null
}

private fun loadStillImageFromArgs(args: Array<String>): File? {
    argValue(args, "--image")?.let { return File(it) }
    System.getenv("SHITSPOTTER_DESKTOP_IMAGE")?.let { return File(it) }
    System.getProperty("desktopHarnessImage")?.let { return File(it) }
    return null
}

private fun chooseDesktopBackend(args: Array<String>): DetectorBackend {
    val modelPath = argValue(args, "--model") ?: System.getenv("SHITSPOTTER_DESKTOP_MODEL")
    val modelId = argValue(args, "--model-id") ?: ModelRegistry.default.modelId
    val spec: ModelSpec = ModelRegistry.byId(modelId) ?: ModelRegistry.default
    if (modelPath.isNullOrBlank()) {
        PrintlnLogger.info("ShitSpotter.Desktop", "no --model — using stub detector")
        return StubDetectorBackend(spec.takeIf { it.format != io.kitware.shitspotter.core.ModelFormat.STUB } ?: ModelSpec.STUB)
    }
    val file = File(modelPath)
    if (!file.isFile) {
        PrintlnLogger.warn("ShitSpotter.Desktop", "model file missing: ${file.absolutePath} — falling back to stub")
        return StubDetectorBackend()
    }
    return try {
        OnnxRuntimeJvmBackend(spec, file.absolutePath).also { it.warmup() }
    } catch (t: Throwable) {
        PrintlnLogger.error("ShitSpotter.Desktop", "ONNX init failed; falling back to stub", t)
        StubDetectorBackend()
    }
}

fun main(args: Array<String>) {
    PrintlnLogger.info("ShitSpotter.Desktop", "starting; commit=${BuildInfo.appCommit}")
    if (runCompareIfRequested(args)) return
    if (runDescribeIfRequested(args)) return

    val state = AppState()
    val settingsStore = FileSettingsStore()
    state.applySettings(settingsStore.load())
    val backend: DetectorBackend = chooseDesktopBackend(args)
    PrintlnLogger.info(
        "ShitSpotter.Desktop",
        "active model = ${state.activeModelId} | backend = ${backend.backendName}" +
            " | delegate = ${backend.delegate ?: "—"} | threshold = ${state.scoreThreshold}",
    )
    val imageFile = loadStillImageFromArgs(args)
    val frameDir = argValue(args, "--frames")?.let { File(it) }
    val frame = imageFile?.takeIf { it.isFile }?.let { StillImageFrameSource.fromFile(it) }
    val frameDirSource = frameDir?.takeIf { it.isDirectory }?.let { FrameDirectorySource(it) }
    val backgroundImage = imageFile?.takeIf { it.isFile }?.let { javax.imageio.ImageIO.read(it) }
    val failureStore = DesktopFailureCaseStore(File("failure_cases"))

    val harness = DesktopHarness(state) { backend }

    application {
        Window(
            onCloseRequest = {
                try { settingsStore.save(state.toSettings()) } catch (_: Throwable) {}
                exitApplication()
            },
            title = "ShitSpotter (desktop harness)",
        ) {
            val scope = rememberCoroutineScope()
            LaunchedEffect(Unit) {
                when {
                    frameDirSource != null -> {
                        PrintlnLogger.info(
                            "ShitSpotter.Desktop",
                            "looping over ${frameDirSource.frameCount} frames in ${frameDir!!.absolutePath}",
                        )
                        harness.runDirectoryLoop(scope, frameDirSource)
                    }
                    frame != null -> {
                        PrintlnLogger.info(
                            "ShitSpotter.Desktop",
                            "looping over still image ${frame.width}x${frame.height}",
                        )
                        harness.runLoop(scope, frame)
                    }
                    else -> {
                        PrintlnLogger.warn(
                            "ShitSpotter.Desktop",
                            "no input; pass --image=<path>, --frames=<dir>, " +
                                "or set SHITSPOTTER_DESKTOP_IMAGE",
                        )
                    }
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
