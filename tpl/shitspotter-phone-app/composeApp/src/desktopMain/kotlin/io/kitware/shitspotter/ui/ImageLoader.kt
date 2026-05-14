package io.kitware.shitspotter.ui

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import java.io.File

actual fun loadImageBitmapFromFile(filePath: String): ImageBitmap? = try {
    val bytes = File(filePath).readBytes()
    org.jetbrains.skia.Image.makeFromEncoded(bytes).toComposeImageBitmap()
} catch (_: Throwable) { null }
