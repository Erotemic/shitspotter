package io.kitware.shitspotter.core

import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 * Tests for `Yolox.decodeRawStrides`, which is the path used when the
 * exported ONNX model emits per-stride raw outputs rather than the
 * already-decoded shape used by the YOLOX-nano poop model. Not
 * currently wired into the live backends — `decodeRawStrides` is a
 * future Milestone-2/3 plug for new model variants — but covered
 * here so the math doesn't bit-rot.
 */
class YoloxRawStridesTest {

    @Test
    fun decode_recovers_pixel_space_for_unit_stride() {
        // 1x1 grid at stride 1 with one anchor; raw delta = 0 → centre
        // should land at (xi*s, yi*s) = (0, 0); raw size = 0 → exp(0)*s = 1.
        val numClasses = 1
        val perRow = 5 + numClasses
        val rows = 1
        val raw = FloatArray(rows * perRow).apply {
            this[0] = 0f  // raw cx delta
            this[1] = 0f  // raw cy delta
            this[2] = 0f  // raw log-w
            this[3] = 0f  // raw log-h
            this[4] = 4f  // obj logit; sigmoid(4) ≈ 0.982
            this[5] = -2f // cls logit; sigmoid(-2) ≈ 0.119
        }
        val decoded = Yolox.decodeRawStrides(
            predictions = raw,
            gridSizes = listOf(1 to 1),
            strides = listOf(1),
            numClasses = numClasses,
        )
        assertEquals(0f, decoded[0])  // cx
        assertEquals(0f, decoded[1])  // cy
        assertEquals(1f, decoded[2])  // w = exp(0) * 1
        assertEquals(1f, decoded[3])  // h
        // Sigmoid(4) and sigmoid(-2) — within floating tolerance.
        assertTrue((decoded[4] - 0.982f) < 0.01f)
        assertTrue((decoded[5] - 0.119f) < 0.01f)
    }

    @Test
    fun decode_handles_multiple_strides() {
        val numClasses = 1
        val perRow = 5 + numClasses
        val grids = listOf(2 to 2, 1 to 1) // 4 + 1 = 5 anchors
        val strides = listOf(8, 16)
        val raw = FloatArray(5 * perRow)
        // The first stride: 2x2 grid at stride 8.
        // anchor (yi=0, xi=0): raw (0,0,0,0) → cx=0, cy=0, w=8, h=8
        // anchor (yi=0, xi=1): raw (0,0,0,0) → cx=8, cy=0, w=8, h=8
        // anchor (yi=1, xi=0): raw (0,0,0,0) → cx=0, cy=8, w=8, h=8
        // anchor (yi=1, xi=1): raw (0,0,0,0) → cx=8, cy=8, w=8, h=8
        // The second stride: 1x1 grid at stride 16.
        // anchor (yi=0, xi=0): raw (0,0,0,0) → cx=0, cy=0, w=16, h=16
        val decoded = Yolox.decodeRawStrides(
            predictions = raw,
            gridSizes = grids,
            strides = strides,
            numClasses = numClasses,
        )
        // Spot check: anchor[1] in first stride is at (cx=8, cy=0).
        assertEquals(8f, decoded[1 * perRow + 0])
        assertEquals(0f, decoded[1 * perRow + 1])
        // Anchor[3] is at (cx=8, cy=8).
        assertEquals(8f, decoded[3 * perRow + 0])
        assertEquals(8f, decoded[3 * perRow + 1])
        // Anchor[4] is the second stride's only cell — w = exp(0)*16 = 16.
        assertEquals(16f, decoded[4 * perRow + 2])
    }

    @Test
    fun mismatched_predictions_size_throws() {
        assertFails {
            Yolox.decodeRawStrides(
                predictions = FloatArray(2),
                gridSizes = listOf(2 to 2),
                strides = listOf(8),
                numClasses = 1,
            )
        }
    }

    @Test
    fun mismatched_grid_strides_throws() {
        assertFails {
            Yolox.decodeRawStrides(
                predictions = FloatArray(0),
                gridSizes = listOf(1 to 1, 2 to 2),
                strides = listOf(8),
                numClasses = 1,
            )
        }
    }
}
