package io.github.erotemic.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
data class FrameTelemetry(
    val deviceModel: String,
    val osVersion: String,
    val appCommit: String,
    val modelId: String,
    val modelHash: String?,
    val runtimeBackend: String,
    val delegate: String?,
    val inputWidth: Int,
    val inputHeight: Int,
    val captureMs: Double,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val overlayMs: Double,
    val fpsRecent: Double,
    val detectionCount: Int,
    val droppedFrames: Long,
) {
    val totalMs: Double
        get() = captureMs + preprocessMs + inferenceMs + postprocessMs + overlayMs
}

class FpsCounter(private val windowMs: Long = 1000L) {
    private val timestamps = ArrayDeque<Long>()

    fun mark(nowMs: Long): Double {
        timestamps.addLast(nowMs)
        val cutoff = nowMs - windowMs
        while (timestamps.isNotEmpty() && timestamps.first() < cutoff) {
            timestamps.removeFirst()
        }
        if (timestamps.size < 2) return 0.0
        val span = timestamps.last() - timestamps.first()
        if (span <= 0L) return 0.0
        return (timestamps.size - 1) * 1000.0 / span
    }

    fun reset() = timestamps.clear()
}

class LatencyAccumulator(private val windowSize: Int = 60) {
    private val samples = ArrayDeque<Double>()

    fun record(ms: Double) {
        samples.addLast(ms)
        while (samples.size > windowSize) samples.removeFirst()
    }

    fun mean(): Double = if (samples.isEmpty()) 0.0 else samples.average()

    fun percentile(p: Double): Double {
        if (samples.isEmpty()) return 0.0
        val sorted = samples.toMutableList().apply { sort() }
        val idx = (sorted.size * p).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    fun reset() = samples.clear()
}
