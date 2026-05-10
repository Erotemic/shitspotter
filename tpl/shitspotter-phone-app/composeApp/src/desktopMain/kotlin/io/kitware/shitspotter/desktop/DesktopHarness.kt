package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FpsCounter
import io.kitware.shitspotter.core.FrameSource
import io.kitware.shitspotter.core.FrameTelemetry
import io.kitware.shitspotter.core.LatencyAccumulator
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.nowMonoMs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.awt.image.BufferedImage

/**
 * The desktop equivalent of [io.kitware.shitspotter.android.CameraAnalysisLoop]:
 * runs the same backend on a still image (or a sequence of frames from a video,
 * once we add a video frame source) and pumps state into the shared [AppState].
 */
class DesktopHarness(
    private val state: AppState,
    private val backendProvider: () -> DetectorBackend,
) {
    private var job: Job? = null
    private val fpsCounter = FpsCounter(windowMs = 1500L)
    private val captureLat = LatencyAccumulator(60)

    fun runOnce(frame: FrameSource) {
        val captureStart = nowMonoMs()
        captureLat.record(nowMonoMs() - captureStart)

        val backend = backendProvider()
        val r = backend.analyze(frame)
        val nowMs = System.currentTimeMillis()
        val fps = fpsCounter.mark(nowMs)

        val tele = FrameTelemetry(
            deviceModel = BuildInfo.deviceModel,
            osVersion = BuildInfo.osVersion,
            appCommit = BuildInfo.appCommit,
            modelId = backend.spec.modelId,
            modelHash = backend.spec.modelHash,
            runtimeBackend = r.backendName,
            delegate = r.delegate,
            inputWidth = backend.spec.inputWidth,
            inputHeight = backend.spec.inputHeight,
            captureMs = captureLat.mean(),
            preprocessMs = r.preprocessMs,
            inferenceMs = r.inferenceMs,
            postprocessMs = r.postprocessMs,
            overlayMs = 0.0,
            fpsRecent = fps,
            detectionCount = r.detections.size,
            droppedFrames = 0L,
        )
        state.pushFrame(r.detections, tele, frame.width, frame.height)
        PrintlnLogger.info(
            "ShitSpotter.Harness",
            "frame ${frame.width}x${frame.height} → ${r.detections.size} dets " +
                "inf=${"%.1f".format(r.inferenceMs)}ms backend=${r.backendName}",
        )
    }

    /** Pretend the still image is a 30 FPS camera, useful for HUD/overlay testing. */
    fun runLoop(scope: CoroutineScope, frame: FrameSource, periodMs: Long = 33L) {
        job?.cancel()
        job = scope.launch(Dispatchers.Default) {
            while (isActive()) {
                runOnce(frame)
                delay(periodMs)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    private fun CoroutineScope.isActive(): Boolean = this.coroutineContext[Job]?.isActive ?: false
}
