package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals

class FmtMsTest {

    @Test
    fun zero() {
        assertEquals("0.0", Fmt.ms(0.0))
    }

    @Test
    fun integer() {
        assertEquals("12.0", Fmt.ms(12.0))
    }

    @Test
    fun rounds_to_one_decimal() {
        assertEquals("1.5", Fmt.ms(1.45))   // banker's rounding could go either way
        assertEquals("1.4", Fmt.ms(1.44))
        assertEquals("1.5", Fmt.ms(1.49))
    }

    @Test
    fun negative_keeps_sign() {
        assertEquals("-12.3", Fmt.ms(-12.3))
    }

    @Test
    fun nan_inf_render_as_dash() {
        assertEquals("—", Fmt.ms(Double.NaN))
        assertEquals("—", Fmt.ms(Double.POSITIVE_INFINITY))
        assertEquals("—", Fmt.ms(Double.NEGATIVE_INFINITY))
    }

    @Test
    fun frac_carry() {
        // 0.95 → 1.0, not 0.10.
        assertEquals("1.0", Fmt.ms(0.95))
    }
}

class FmtScoreTest {

    @Test
    fun three_decimal_places() {
        assertEquals("0.987", Fmt.score(0.987f))
        assertEquals("0.001", Fmt.score(0.001f))
        assertEquals("0.500", Fmt.score(0.5f))
    }

    @Test
    fun zero_pads() {
        assertEquals("0.050", Fmt.score(0.05f))
        assertEquals("0.005", Fmt.score(0.005f))
    }

    @Test
    fun integer_part() {
        assertEquals("1.000", Fmt.score(1.0f))
        assertEquals("12.345", Fmt.score(12.345f))
    }

    @Test
    fun negative_keeps_sign() {
        assertEquals("-0.500", Fmt.score(-0.5f))
    }

    @Test
    fun nan_renders_as_dash() {
        assertEquals("—", Fmt.score(Float.NaN))
    }

    @Test
    fun rollover_carries() {
        // 0.9999 → 1.000, not 0.1000.
        assertEquals("1.000", Fmt.score(0.9999f))
    }
}
