package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FilterByScoreTest {
    private val a = Detection(BoundingBox(0f, 0f, 1f, 1f), score = 0.10f)
    private val b = Detection(BoundingBox(0f, 0f, 1f, 1f), score = 0.50f)
    private val c = Detection(BoundingBox(0f, 0f, 1f, 1f), score = 0.90f)

    @Test
    fun zero_threshold_keeps_everything() {
        val out = listOf(a, b, c).filterByScore(0f)
        assertEquals(3, out.size)
    }

    @Test
    fun high_threshold_drops_low_scores() {
        val out = listOf(a, b, c).filterByScore(0.4f)
        assertEquals(2, out.size)
        assertTrue(out.all { it.score >= 0.4f })
    }

    @Test
    fun threshold_above_max_returns_empty() {
        assertEquals(0, listOf(a, b, c).filterByScore(0.99f).size)
    }
}
