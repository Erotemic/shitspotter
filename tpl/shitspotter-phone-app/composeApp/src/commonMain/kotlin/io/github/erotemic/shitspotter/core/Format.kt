package io.github.erotemic.shitspotter.core

import kotlin.math.abs
import kotlin.math.roundToLong

/**
 * Tiny multiplatform format helpers. Kotlin's String.format is JVM-only,
 * and we want the same UI numbers to render identically across the
 * Android app and the desktop harness. These are intentionally narrow:
 * one decimal place for ms / FPS, three for scores. If you need more,
 * write a new helper rather than parameterising these.
 */
object Fmt {
    /** Format a Double to one decimal place. NaN/Inf → "—". */
    fun ms(v: Double): String {
        if (v.isNaN() || v.isInfinite()) return "—"
        val sign = if (v < 0) "-" else ""
        val abs = abs(v)
        val whole = abs.toLong()
        val frac = ((abs - whole) * 10).roundToLong()
        val (w, f) = if (frac >= 10) (whole + 1) to 0L else whole to frac
        return "$sign$w.$f"
    }

    /** Format a Double to two decimal places. NaN/Inf → "—". */
    fun ms2(v: Double): String {
        if (v.isNaN() || v.isInfinite()) return "—"
        val sign = if (v < 0) "-" else ""
        val abs = abs(v)
        val whole = abs.toLong()
        val frac = ((abs - whole) * 100.0 + 0.5).toLong()
        val (w, f) = if (frac >= 100) (whole + 1) to 0L else whole to frac
        val fStr = if (f < 10) "0$f" else f.toString()
        return "$sign$w.$fStr"
    }

    /** Format a Float score to three decimal places (no sign for >=0). */
    fun score(v: Float): String {
        if (v.isNaN() || v.isInfinite()) return "—"
        val sign = if (v < 0) "-" else ""
        val abs = if (v < 0) -v else v
        val whole = abs.toInt()
        val frac = ((abs - whole) * 1000.0f + 0.5f).toInt()
        val (w, f) = if (frac >= 1000) (whole + 1) to 0 else whole to frac
        val fStr = when {
            f >= 100 -> f.toString()
            f >= 10 -> "0$f"
            else -> "00$f"
        }
        return "$sign$w.$fStr"
    }
}
