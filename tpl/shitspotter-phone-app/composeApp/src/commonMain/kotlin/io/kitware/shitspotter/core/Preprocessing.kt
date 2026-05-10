package io.kitware.shitspotter.core

/**
 * Shared CPU-side image preprocessing. Backends may bypass this for
 * hardware-accelerated paths, but having a reference implementation in
 * commonMain means the desktop harness and Android stub take the same
 * path, which makes regression testing far easier.
 *
 * Input is RGB888 (interleaved, top-left origin), output is FloatArray
 * laid out per [ModelSpec.inputLayout] (NCHW or NHWC) using the spec's
 * normalization.
 */
object Preprocessing {

    /**
     * Stretch [rgb] from [srcW]x[srcH] to [dstW]x[dstH] using nearest-
     * neighbour sampling. Returns a fake LetterboxParams with scale equal
     * to the average horizontal/vertical scale and zero padding so the
     * caller can re-use the same letterbox-undo math (boxes will be
     * slightly distorted on aspect-ratio mismatch — that's the trade-off
     * for STRETCH mode).
     */
    fun stretchRgb(
        rgb: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
    ): Pair<ByteArray, LetterboxParams> {
        require(rgb.size == srcW * srcH * 3) {
            "rgb buffer size ${rgb.size} != ${srcW * srcH * 3}"
        }
        val out = ByteArray(dstW * dstH * 3)
        val sx = srcW.toFloat() / dstW
        val sy = srcH.toFloat() / dstH
        var dst = 0
        for (y in 0 until dstH) {
            val sYi = (y * sy).toInt().coerceIn(0, srcH - 1)
            for (x in 0 until dstW) {
                val sXi = (x * sx).toInt().coerceIn(0, srcW - 1)
                val srcOff = (sYi * srcW + sXi) * 3
                out[dst] = rgb[srcOff]
                out[dst + 1] = rgb[srcOff + 1]
                out[dst + 2] = rgb[srcOff + 2]
                dst += 3
            }
        }
        // STRETCH does not preserve aspect ratio; return a synthetic
        // LetterboxParams that the caller can use to undo the scaling.
        // Boxes from the model will need both x and y scales separately
        // (we pick the geometric mean here for a single-scale undo, which
        // is correct only when src and dst have matching aspect ratios).
        val params = LetterboxParams(
            scale = (dstW.toFloat() / srcW + dstH.toFloat() / srcH) / 2f,
            padX = 0f,
            padY = 0f,
            outWidth = dstW,
            outHeight = dstH,
            srcWidth = srcW,
            srcHeight = srcH,
        )
        return out to params
    }

    /**
     * Center-crop [rgb] to [dstW]x[dstH] aspect ratio (whichever the source
     * is taller in), then nearest-neighbour scale to exactly [dstW]x[dstH].
     * Preserves aspect ratio at the cost of cropping the edges.
     */
    fun centerCropRgb(
        rgb: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
    ): Pair<ByteArray, LetterboxParams> {
        require(rgb.size == srcW * srcH * 3) {
            "rgb buffer size ${rgb.size} != ${srcW * srcH * 3}"
        }
        val targetAspect = dstW.toFloat() / dstH
        val srcAspect = srcW.toFloat() / srcH
        val (cropW, cropH) = if (srcAspect > targetAspect) {
            // src is wider than target — crop the sides.
            val w = (srcH * targetAspect).toInt().coerceAtLeast(1)
            w to srcH
        } else {
            // src is taller — crop the top/bottom.
            val h = (srcW / targetAspect).toInt().coerceAtLeast(1)
            srcW to h
        }
        val cropX = (srcW - cropW) / 2
        val cropY = (srcH - cropH) / 2
        val cropped = ByteArray(cropW * cropH * 3)
        for (y in 0 until cropH) {
            val srcRow = (cropY + y) * srcW + cropX
            val dstRow = y * cropW
            for (x in 0 until cropW) {
                val s = (srcRow + x) * 3
                val d = (dstRow + x) * 3
                cropped[d] = rgb[s]
                cropped[d + 1] = rgb[s + 1]
                cropped[d + 2] = rgb[s + 2]
            }
        }
        val (stretched, _) = stretchRgb(cropped, cropW, cropH, dstW, dstH)
        // The single-scale LetterboxParams needs both crop offsets and
        // the stretch scale. We model the crop as negative padding
        // because mapBoxToSource computes (x - padX)/scale to recover
        // src-space. For center-crop, src-space x = (dst-space x)/scale +
        // cropX, so padX = -cropX*scale gives the right round-trip
        // formula when scale = dstW/cropW.
        val scaleX = dstW.toFloat() / cropW
        val scaleY = dstH.toFloat() / cropH
        val scale = (scaleX + scaleY) / 2f
        return stretched to LetterboxParams(
            scale = scale,
            padX = -cropX.toFloat() * scale,
            padY = -cropY.toFloat() * scale,
            outWidth = dstW,
            outHeight = dstH,
            srcWidth = srcW,
            srcHeight = srcH,
        )
    }

    fun letterboxRgb(
        rgb: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
        padValue: Int = 114,
    ): Pair<ByteArray, LetterboxParams> {
        require(rgb.size == srcW * srcH * 3) {
            "rgb buffer size ${rgb.size} != ${srcW * srcH * 3}"
        }
        val params = LetterboxParams.compute(srcW, srcH, dstW, dstH)
        val newW = (srcW * params.scale).toInt().coerceAtLeast(1)
        val newH = (srcH * params.scale).toInt().coerceAtLeast(1)
        val out = ByteArray(dstW * dstH * 3) { padValue.toByte() }

        // Nearest-neighbor for speed; backends may swap in bilinear.
        val padX = params.padX.toInt()
        val padY = params.padY.toInt()
        for (y in 0 until newH) {
            val srcY = (y / params.scale).toInt().coerceIn(0, srcH - 1)
            val outY = padY + y
            if (outY < 0 || outY >= dstH) continue
            for (x in 0 until newW) {
                val srcX = (x / params.scale).toInt().coerceIn(0, srcW - 1)
                val outX = padX + x
                if (outX < 0 || outX >= dstW) continue
                val srcOff = (srcY * srcW + srcX) * 3
                val dstOff = (outY * dstW + outX) * 3
                out[dstOff] = rgb[srcOff]
                out[dstOff + 1] = rgb[srcOff + 1]
                out[dstOff + 2] = rgb[srcOff + 2]
            }
        }
        return out to params
    }

    fun toFloatTensor(
        rgb: ByteArray,
        width: Int,
        height: Int,
        layout: InputLayout,
        colorOrder: ColorOrder,
        normalization: Normalization,
    ): FloatArray {
        require(rgb.size == width * height * 3) {
            "rgb buffer size ${rgb.size} != ${width * height * 3}"
        }
        val total = width * height * 3
        val out = FloatArray(total)
        val mean = normalization.mean
        val std = normalization.std
        val scale = normalization.scale
        val swapRb = colorOrder == ColorOrder.BGR
        when (layout) {
            InputLayout.NHWC -> {
                var i = 0
                while (i < total) {
                    val r = (rgb[i].toInt() and 0xFF) * scale
                    val g = (rgb[i + 1].toInt() and 0xFF) * scale
                    val b = (rgb[i + 2].toInt() and 0xFF) * scale
                    if (swapRb) {
                        out[i] = (b - mean[0]) / std[0]
                        out[i + 1] = (g - mean[1]) / std[1]
                        out[i + 2] = (r - mean[2]) / std[2]
                    } else {
                        out[i] = (r - mean[0]) / std[0]
                        out[i + 1] = (g - mean[1]) / std[1]
                        out[i + 2] = (b - mean[2]) / std[2]
                    }
                    i += 3
                }
            }
            InputLayout.NCHW -> {
                val plane = width * height
                var px = 0
                var i = 0
                while (i < total) {
                    val r = (rgb[i].toInt() and 0xFF) * scale
                    val g = (rgb[i + 1].toInt() and 0xFF) * scale
                    val b = (rgb[i + 2].toInt() and 0xFF) * scale
                    val c0 = if (swapRb) b else r
                    val c2 = if (swapRb) r else b
                    out[px] = (c0 - mean[0]) / std[0]
                    out[plane + px] = (g - mean[1]) / std[1]
                    out[2 * plane + px] = (c2 - mean[2]) / std[2]
                    px++
                    i += 3
                }
            }
        }
        return out
    }
}
