package io.kitware.shitspotter.ui

import android.graphics.BitmapFactory
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap

actual fun loadImageBitmapFromFile(filePath: String): ImageBitmap? = try {
    BitmapFactory.decodeFile(filePath)?.asImageBitmap()
} catch (_: Throwable) { null }
