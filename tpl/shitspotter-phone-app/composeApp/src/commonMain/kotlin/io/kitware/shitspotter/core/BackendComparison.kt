package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

/**
 * Milestone-3 backend comparison primitive: runs the same frame through
 * multiple backends and records timings + detections. The shared layer
 * does not know which backends are real — that is supplied by the
 * platform layer (Android can pass NNAPI vs CPU, desktop can pass CPU
 * only or CUDA EP later).
 *
 * The output is a small data class that can be JSON-serialized and
 * dumped next to a benchmark report, keeping the per-frame schema
 * platform-agnostic.
 */
@Serializable
data class BackendRunRow(
    val frameWidth: Int,
    val frameHeight: Int,
    val backendName: String,
    val delegate: String?,
    val modelId: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val preprocessMs: Double,
    val inferenceMs: Double,
    val postprocessMs: Double,
    val detectionCount: Int,
    val topScore: Float,
)

@Serializable
data class BackendComparisonReport(
    val timestamp: String,
    val deviceModel: String,
    val osVersion: String,
    val rows: List<BackendRunRow>,
)

object BackendComparison {

    /**
     * Run [frame] through every backend in [backends] [warmupRuns] +
     * [measureRuns] times. Returns the per-row mean of the measured runs
     * (warmup runs are dropped). Each backend's [DetectorBackend.warmup]
     * is also called once before timing starts.
     */
    fun runMeasured(
        frame: FrameSource,
        backends: List<DetectorBackend>,
        warmupRuns: Int = 1,
        measureRuns: Int = 5,
    ): List<BackendRunRow> = backends.map { backend ->
        backend.warmup()
        repeat(warmupRuns) { backend.analyze(frame) }
        var pre = 0.0
        var inf = 0.0
        var post = 0.0
        var dets = 0
        var topScore = 0f
        repeat(measureRuns) {
            val r = backend.analyze(frame)
            pre += r.preprocessMs
            inf += r.inferenceMs
            post += r.postprocessMs
            dets = r.detections.size
            topScore = r.detections.maxOfOrNull { it.score } ?: 0f
        }
        BackendRunRow(
            frameWidth = frame.width,
            frameHeight = frame.height,
            backendName = backend.backendName,
            delegate = backend.delegate,
            modelId = backend.spec.modelId,
            inputWidth = backend.spec.inputWidth,
            inputHeight = backend.spec.inputHeight,
            preprocessMs = pre / measureRuns,
            inferenceMs = inf / measureRuns,
            postprocessMs = post / measureRuns,
            detectionCount = dets,
            topScore = topScore,
        )
    }

    fun renderTable(rows: List<BackendRunRow>): String {
        if (rows.isEmpty()) return "(no rows)"
        val header = listOf(
            padRight("model", 32),
            padRight("backend", 22),
            padRight("delegate", 8),
            padLeft("pre(ms)", 8),
            padLeft("inf(ms)", 8),
            padLeft("post(ms)", 8),
            padLeft("dets", 5),
            padLeft("top", 7),
        ).joinToString(" | ")
        val sep = "-".repeat(header.length)
        val body = rows.joinToString("\n") { r ->
            listOf(
                padRight(r.modelId.take(32), 32),
                padRight(r.backendName.take(22), 22),
                padRight((r.delegate ?: "—").take(8), 8),
                padLeft(formatMs(r.preprocessMs), 8),
                padLeft(formatMs(r.inferenceMs), 8),
                padLeft(formatMs(r.postprocessMs), 8),
                padLeft(r.detectionCount.toString(), 5),
                padLeft(formatScore(r.topScore), 7),
            ).joinToString(" | ")
        }
        return "$header\n$sep\n$body"
    }

    private fun padRight(s: String, n: Int): String =
        if (s.length >= n) s else s + " ".repeat(n - s.length)

    private fun padLeft(s: String, n: Int): String =
        if (s.length >= n) s else " ".repeat(n - s.length) + s

    private fun formatMs(v: Double): String = Fmt.ms2(v)

    private fun formatScore(v: Float): String = Fmt.score(v)
}
