package io.kitware.shitspotter.android

import android.content.Context
import android.graphics.ImageFormat
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import io.kitware.shitspotter.core.AppLogger
import io.kitware.shitspotter.core.AppState
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FpsCounter
import io.kitware.shitspotter.core.FrameSource
import io.kitware.shitspotter.core.FrameTelemetry
import io.kitware.shitspotter.core.LatencyAccumulator
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.nowMonoMs
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * Wires CameraX [ImageAnalysis] into a [DetectorBackend]. KEEP_ONLY_LATEST is
 * the only acceptable strategy here — we never queue stale frames behind a
 * slow detector. Every ImageProxy is closed exactly once.
 */
class CameraAnalysisLoop(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val state: AppState,
    private val backendProvider: () -> DetectorBackend,
    private val logger: AppLogger = PrintlnLogger,
    private val targetAnalysisSize: Size = Size(640, 480),
) {
    @Volatile var isPaused: Boolean = false
    @Volatile private var droppedFrames: AtomicLong = AtomicLong(0L)

    private val analyzerExecutor: Executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "shitspotter-analyzer").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val fpsCounter = FpsCounter(windowMs = 1500L)
    private val captureLat = LatencyAccumulator(60)
    private val preLat = LatencyAccumulator(60)
    private val infLat = LatencyAccumulator(60)
    private val postLat = LatencyAccumulator(60)
    private val overlayLat = LatencyAccumulator(60)

    fun bind(
        previewSurfaceProvider: Preview.SurfaceProvider,
    ): ListenableFuture<ProcessCameraProvider> {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewSurfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    targetAnalysisSize,
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                ),
                            )
                            .build(),
                    )
                    .build()
                analysis.setAnalyzer(analyzerExecutor) { proxy -> handleFrame(proxy) }

                val selector = CameraSelector.DEFAULT_BACK_CAMERA
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
                logger.info(TAG, "CameraX bound; analysis target=$targetAnalysisSize")
            } catch (t: Throwable) {
                logger.error(TAG, "CameraX bind failed", t)
                state.setError("Camera bind failed: ${t.message}")
            }
        }, ContextCompatExecutors.main(context))
        return future
    }

    private fun handleFrame(proxy: ImageProxy) {
        try {
            if (isPaused) {
                droppedFrames.incrementAndGet()
                return
            }
            val captureStart = nowMonoMs()
            val frame = ImageProxyFrame.from(proxy)
            captureLat.record(nowMonoMs() - captureStart)

            val backend = backendProvider()
            val result = backend.analyze(frame)

            preLat.record(result.preprocessMs)
            infLat.record(result.inferenceMs)
            postLat.record(result.postprocessMs)

            val overlayStart = nowMonoMs()
            val nowMs = System.currentTimeMillis()
            val fps = fpsCounter.mark(nowMs)
            val overlayMs = nowMonoMs() - overlayStart
            overlayLat.record(overlayMs)

            val telemetry = FrameTelemetry(
                deviceModel = BuildInfo.deviceModel,
                osVersion = BuildInfo.osVersion,
                appCommit = BuildInfo.appCommit,
                modelId = backend.spec.modelId,
                modelHash = backend.spec.modelHash,
                runtimeBackend = result.backendName,
                delegate = result.delegate,
                inputWidth = backend.spec.inputWidth,
                inputHeight = backend.spec.inputHeight,
                captureMs = captureLat.mean(),
                preprocessMs = result.preprocessMs,
                inferenceMs = result.inferenceMs,
                postprocessMs = result.postprocessMs,
                overlayMs = overlayMs,
                fpsRecent = fps,
                detectionCount = result.detections.size,
                droppedFrames = droppedFrames.get(),
            )
            state.pushFrame(result.detections, telemetry, frame.width, frame.height)
        } catch (t: Throwable) {
            logger.error(TAG, "frame analysis failed", t)
            state.setError(t.message ?: t::class.simpleName ?: "unknown")
        } finally {
            proxy.close()
        }
    }

    companion object {
        const val TAG = "ShitSpotter.AnalysisLoop"
    }
}

/** Wraps a CameraX ImageProxy as a [FrameSource]. We only convert to RGB888
 * lazily — backends that don't need RGB skip the copy. */
class ImageProxyFrame private constructor(
    override val width: Int,
    override val height: Int,
    override val rotationDegrees: Int,
    private val rgbaPlane: ByteArray,
    private val rowStride: Int,
    private val pixelStride: Int,
) : FrameSource {

    override fun toRgb888(): ByteArray {
        if (rowStride == width * pixelStride && pixelStride == 4) {
            // Tight RGBA8888 — reduce in place.
            val out = ByteArray(width * height * 3)
            var src = 0
            var dst = 0
            val total = width * height
            for (i in 0 until total) {
                out[dst] = rgbaPlane[src]
                out[dst + 1] = rgbaPlane[src + 1]
                out[dst + 2] = rgbaPlane[src + 2]
                src += 4
                dst += 3
            }
            return out
        }
        // Fallback: row-by-row stride-aware copy.
        val out = ByteArray(width * height * 3)
        var dst = 0
        for (y in 0 until height) {
            val rowStart = y * rowStride
            for (x in 0 until width) {
                val src = rowStart + x * pixelStride
                out[dst] = rgbaPlane[src]
                out[dst + 1] = rgbaPlane[src + 1]
                out[dst + 2] = rgbaPlane[src + 2]
                dst += 3
            }
        }
        return out
    }

    companion object {
        fun from(proxy: ImageProxy): ImageProxyFrame {
            val plane = proxy.planes[0]
            val buf = plane.buffer
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            return ImageProxyFrame(
                width = proxy.width,
                height = proxy.height,
                rotationDegrees = proxy.imageInfo.rotationDegrees,
                rgbaPlane = bytes,
                rowStride = plane.rowStride,
                pixelStride = plane.pixelStride,
            )
        }
    }
}

/** Tiny shim around ContextCompat.getMainExecutor with a stable name. */
internal object ContextCompatExecutors {
    fun main(context: Context): Executor =
        androidx.core.content.ContextCompat.getMainExecutor(context)
}
