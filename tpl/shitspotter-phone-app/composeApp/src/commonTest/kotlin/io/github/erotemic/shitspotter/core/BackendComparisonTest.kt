package io.github.erotemic.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BackendComparisonTest {

    private class FakeFrame : FrameSource {
        override val width = 320
        override val height = 240
        override val rotationDegrees = 0
        override fun toRgb888(): ByteArray = ByteArray(width * height * 3)
    }

    @Test
    fun renderTable_handles_empty() {
        assertEquals("(no rows)", BackendComparison.renderTable(emptyList()))
    }

    @Test
    fun runMeasured_zero_runs_throws() {
        var threw = false
        try {
            BackendComparison.runMeasured(
                FakeFrame(),
                listOf(StubDetectorBackend()),
                warmupRuns = 0,
                measureRuns = 0,
            )
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertEquals(true, threw)
    }

    @Test
    fun runMeasured_negative_warmup_throws() {
        var threw = false
        try {
            BackendComparison.runMeasured(
                FakeFrame(),
                listOf(StubDetectorBackend()),
                warmupRuns = -1,
                measureRuns = 1,
            )
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertEquals(true, threw)
    }

    @Test
    fun runMeasured_produces_one_row_per_backend() {
        val a = StubDetectorBackend(ModelSpec.STUB.copy(modelId = "stub-a"))
        val b = StubDetectorBackend(ModelSpec.STUB.copy(modelId = "stub-b"))
        val rows = BackendComparison.runMeasured(
            frame = FakeFrame(),
            backends = listOf(a, b),
            warmupRuns = 1,
            measureRuns = 3,
        )
        assertEquals(2, rows.size)
        assertEquals("stub-a", rows[0].modelId)
        assertEquals("stub-b", rows[1].modelId)
        rows.forEach {
            assertTrue(it.preprocessMs >= 0.0)
            assertTrue(it.inferenceMs >= 0.0)
            assertTrue(it.postprocessMs >= 0.0)
        }
    }

    @Test
    fun renderTable_aligns_columns() {
        val rows = listOf(
            BackendRunRow(
                frameWidth = 640, frameHeight = 480,
                backendName = "onnx-cpu", delegate = "CPU",
                modelId = "stub", inputWidth = 416, inputHeight = 416,
                preprocessMs = 1.234, inferenceMs = 5.6, postprocessMs = 0.05,
                detectionCount = 2, topScore = 0.987f,
            ),
        )
        val table = BackendComparison.renderTable(rows)
        assertTrue(table.contains("onnx-cpu"))
        assertTrue(table.contains("CPU"))
        assertTrue(table.contains("0.987") || table.contains("0.988"))
    }
}
