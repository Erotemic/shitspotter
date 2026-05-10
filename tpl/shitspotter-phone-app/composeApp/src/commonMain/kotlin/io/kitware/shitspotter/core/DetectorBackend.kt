package io.kitware.shitspotter.core

interface FrameSource {
    val width: Int
    val height: Int
    val rotationDegrees: Int
    fun toRgb888(): ByteArray
}

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
        val n = (counter++ % 600L)
        val w = frame.width.toFloat()
        val h = frame.height.toFloat()
        val cx = w * (0.4f + 0.2f * sineApprox(n / 60.0))
        val cy = h * (0.5f + 0.15f * cosineApprox(n / 45.0))
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
        // Suppress unused locals — preprocessMs above is real timing.
        @Suppress("UNUSED_VARIABLE")
        val rgbLen = rgb.size
        val inferenceMs = nowMonoMs() - infStart
        val post = 0.0
        return InferenceResult(
            detections = listOf(det),
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = post,
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
