package io.kitware.shitspotter.ui

import androidx.compose.ui.graphics.ImageBitmap

expect fun loadImageBitmapFromFile(filePath: String): ImageBitmap?
