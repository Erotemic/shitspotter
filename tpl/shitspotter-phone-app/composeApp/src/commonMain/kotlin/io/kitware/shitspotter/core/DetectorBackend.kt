package io.kitware.shitspotter.core

interface FrameSource {
    val width: Int
    val height: Int
    val rotationDegrees: Int
    fun toRgb888(): ByteArray
}

/**
 * Hard floor used by every real DetectorBackend's score filter. The
 * `ModelSpec.scoreThreshold` is the *default UI threshold* (initial
 * slider value); the backend always passes this floor to its
 * postprocess so the UI slider can recover detections all the way
 * down to 1%. Below this floor the cost of NMS over thousands of
 * near-zero anchors isn't worth it.
 *
 * Two explicit thresholds:
 *   - BACKEND_FLOOR_THRESHOLD (this value, fixed) — what the backend
 *     filters at before returning detections to the analysis loop.
 *   - state.scoreThreshold (mutable, UI-controlled) — what the
 *     analysis loop filters at before pushing into AppState.
 */
const val BACKEND_FLOOR_THRESHOLD: Float = 0.01f

data class InferenceResult(
    val detections: List<Detection>,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val backendName: String,
    val delegate: String?,
)

interface DetectorBackend : AutoCloseable {
    val backendName: String
    val delegate: String?
    val spec: ModelSpec

    fun warmup()
    fun analyze(frame: FrameSource): InferenceResult
    override fun close()
}

class StubDetectorBackend(
    override val spec: ModelSpec = ModelSpec.STUB,
) : DetectorBackend {
    override val backendName: String = "stub-1.0"
    override val delegate: String? = null
    private var counter: Long = 0L
    private var closed = false

    override fun warmup() = Unit

    override fun analyze(frame: FrameSource): InferenceResult {
        check(!closed) { "Backend already closed" }
        val pre = nowMonoMs()
        val rgb = frame.toRgb888()
        val preprocessMs = nowMonoMs() - pre

        val infStart = nowMonoMs()
        @Suppress("UNUSED_VARIABLE")
        val rgbLen = rgb.size

        // Camera-only or any no-class stub — skip fake detection generation entirely.
        if (spec.classNames.isEmpty()) {
            val inferenceMs = nowMonoMs() - infStart
            return InferenceResult(
                detections = emptyList(),
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                postprocessMs = 0.0,
                backendName = backendName,
                delegate = delegate,
            )
        }

        val n = (counter++ % 600L)
        val w = frame.width.toFloat()
        val h = frame.height.toFloat()
        val cx = w * (0.4f + 0.2f * sineApprox(n / 60.0).toFloat())
        val cy = h * (0.5f + 0.15f * cosineApprox(n / 45.0).toFloat())
        val bw = w * 0.18f
        val bh = h * 0.22f
        val fakeBox = BoundingBox(
            x = (cx - bw / 2f).coerceAtLeast(0f),
            y = (cy - bh / 2f).coerceAtLeast(0f),
            width = bw.coerceAtMost(w),
            height = bh.coerceAtMost(h),
        )
        val score = 0.5f + 0.4f * sineApprox(n / 30.0).toFloat()
        val det = Detection(box = fakeBox, score = score, classId = 0, className = spec.classNames.firstOrNull())
        val inferenceMs = nowMonoMs() - infStart
        return InferenceResult(
            detections = listOf(det),
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = 0.0,
            backendName = backendName,
            delegate = delegate,
        )
    }

    override fun close() {
        closed = true
    }

    private fun sineApprox(t: Double): Double {
        val twoPi = 6.283185307179586
        val x = ((t % 1.0) + 1.0) % 1.0
        val theta = x * twoPi
        // Bhaskara I approximation, accurate enough for fake-box jitter.
        val pi = 3.141592653589793
        val a = 16.0 * theta * (pi - theta)
        val b = 5.0 * pi * pi - 4.0 * theta * (pi - theta)
        val s = if (theta <= pi) a / b else -a / b
        return s
    }

    private fun cosineApprox(t: Double): Double = sineApprox(t + 0.25)
}

expect fun nowMonoMs(): Double
