package io.github.erotemic.shitspotter.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.exifinterface.media.ExifInterface

actual fun loadImageBitmapFromFile(filePath: String): ImageBitmap? {
    return try {
    // Read dimensions first to compute the right sample size without loading pixels.
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(filePath, bounds)
    val srcW = bounds.outWidth; val srcH = bounds.outHeight
    if (srcW <= 0 || srcH <= 0) return null

    // Subsample so the longest side is ≤ 1280 px — plenty for any phone screen
    // and 8–16× faster to decode than a full 12 MP frame.
    var sampleSize = 1
    while (maxOf(srcW, srcH) / sampleSize > 1280) sampleSize *= 2

    val raw = BitmapFactory.decodeFile(filePath, BitmapFactory.Options().apply {
        inSampleSize = sampleSize
    }) ?: return null

    // Apply EXIF orientation so the image appears right-side-up.
    val degrees = when (
        ExifInterface(filePath).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
    ) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
    }
    val bitmap = if (degrees != 0f) {
        val m = Matrix().apply { postRotate(degrees) }
        Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            .also { if (it !== raw) raw.recycle() }
    } else raw

        bitmap.asImageBitmap()
    } catch (_: Throwable) { null }
}
