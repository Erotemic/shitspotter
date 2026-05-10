package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails

class BoundingBoxRotationTest {
    private val box = BoundingBox(x = 10f, y = 20f, width = 30f, height = 40f)
    private val w = 200
    private val h = 100

    @Test
    fun zero_rotation_is_identity() {
        val (b, nw, nh) = box.rotated(0, w, h)
        assertEquals(box, b)
        assertEquals(w, nw)
        assertEquals(h, nh)
    }

    @Test
    fun ninety_swaps_axes() {
        val (b, nw, nh) = box.rotated(90, w, h)
        assertEquals(h, nw)
        assertEquals(w, nh)
        // Original (10, 20, 30, 40) rotated 90° clockwise within a 200x100 frame:
        //   new x = frameH - (y+height) = 100 - 60 = 40
        //   new y = x = 10
        //   new w = h = 40
        //   new h = w = 30
        assertEquals(40f, b.x)
        assertEquals(10f, b.y)
        assertEquals(40f, b.width)
        assertEquals(30f, b.height)
    }

    @Test
    fun one_eighty_inverts_origin() {
        val (b, nw, nh) = box.rotated(180, w, h)
        assertEquals(w, nw)
        assertEquals(h, nh)
        assertEquals(160f, b.x)  // 200 - (10+30)
        assertEquals(40f, b.y)   //  100 - (20+40)
        assertEquals(box.width, b.width)
        assertEquals(box.height, b.height)
    }

    @Test
    fun two_seventy_swaps_axes() {
        val (b, nw, nh) = box.rotated(270, w, h)
        assertEquals(h, nw)
        assertEquals(w, nh)
        // new x = y = 20
        // new y = frameW - (x+width) = 200 - 40 = 160
        assertEquals(20f, b.x)
        assertEquals(160f, b.y)
        assertEquals(40f, b.width)
        assertEquals(30f, b.height)
    }

    @Test
    fun rotation_normalizes_negative_and_overshoot() {
        val (b1, _, _) = box.rotated(-90, w, h)
        val (b2, _, _) = box.rotated(270, w, h)
        assertEquals(b2, b1)
        val (b3, _, _) = box.rotated(450, w, h)
        val (b4, _, _) = box.rotated(90, w, h)
        assertEquals(b4, b3)
    }

    @Test
    fun rotation_must_be_multiple_of_90() {
        assertFails { box.rotated(45, w, h) }
    }

    @Test
    fun double_rotation_returns_to_original_for_180() {
        val (b1, w1, h1) = box.rotated(180, w, h)
        val (b2, w2, h2) = b1.rotated(180, w1, h1)
        assertEquals(box, b2)
        assertEquals(w, w2)
        assertEquals(h, h2)
    }

    @Test
    fun four_quarter_rotations_return_to_original() {
        var b = box
        var W = w
        var H = h
        repeat(4) {
            val (nb, nw, nh) = b.rotated(90, W, H)
            b = nb; W = nw; H = nh
        }
        assertEquals(box, b)
        assertEquals(w, W)
        assertEquals(h, H)
    }
}
