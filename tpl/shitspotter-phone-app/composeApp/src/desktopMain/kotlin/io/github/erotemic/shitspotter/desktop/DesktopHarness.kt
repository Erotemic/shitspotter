package io.github.erotemic.shitspotter.desktop

import io.github.erotemic.shitspotter.core.AppState
import io.github.erotemic.shitspotter.core.BuildInfo
import io.github.erotemic.shitspotter.core.DetectorBackend
import io.github.erotemic.shitspotter.core.FpsCounter
import io.github.erotemic.shitspotter.core.FrameSource
import io.github.erotemic.shitspotter.core.FrameTelemetry
import io.github.erotemic.shitspotter.core.LatencyAccumulator
import io.github.erotemic.shitspotter.core.filterByScore
import io.github.erotemic.shitspotter.core.PrintlnLogger
import io.github.erotemic.shitspotter.core.nowMonoMs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.awt.image.BufferedImage

/**
 * The desktop equivalent of [io.github.erotemic.shitspotter.android.CameraAnalysisLoop]:
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

        val filtered = r.detections.filterByScore(state.scoreThreshold)
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
            detectionCount = filtered.size,
            droppedFrames = 0L,
        )
        state.pushFrame(filtered, tele, frame.width, frame.height)
        PrintlnLogger.info(
            "ShitSpotter.Harness",
            "frame ${frame.width}x${frame.height} → ${filtered.size}/${r.detections.size} dets " +
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

    /** Replay a directory of frames as a synthetic 30 FPS feed. */
    fun runDirectoryLoop(scope: CoroutineScope, source: FrameDirectorySource, periodMs: Long = 33L) {
        job?.cancel()
        job = scope.launch(Dispatchers.Default) {
            while (isActive()) {
                runOnce(source.next())
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
