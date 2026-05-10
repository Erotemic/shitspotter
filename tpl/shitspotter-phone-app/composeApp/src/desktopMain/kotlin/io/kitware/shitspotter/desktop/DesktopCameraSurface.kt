package io.kitware.shitspotter.desktop

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.unit.sp
import io.kitware.shitspotter.ui.CameraSurface
import io.kitware.shitspotter.ui.OverlayScaleMode
import org.jetbrains.skia.Image as SkiaImage
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

/**
 * Desktop "camera": just renders the still image being analyzed. Optional
 * overlay still comes from the shared [DetectionOverlay] composable so we
 * exercise the same overlay math used on Android.
 */
class DesktopCameraSurface(
    private val backgroundImage: BufferedImage?,
) : CameraSurface {
    private val cached: ImageBitmap? = backgroundImage?.let { bufferedImageToImageBitmap(it) }

    /** Desktop draws the still image with the same min-scale letterbox
     *  that PreviewView FIT_CENTER would use, so the overlay must too. */
    override val overlayScaleMode: OverlayScaleMode = OverlayScaleMode.FIT_CENTER

    @Composable
    override fun Render(modifier: Modifier) {
        Box(modifier = modifier.background(Color(0xFF0A0A0A))) {
            if (cached != null) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = cached.width.toFloat()
                    val h = cached.height.toFloat()
                    val scale = minOf(size.width / w, size.height / h)
                    val drawnW = w * scale
                    val drawnH = h * scale
                    val offX = (size.width - drawnW) / 2f
                    val offY = (size.height - drawnH) / 2f
                    drawImage(
                        image = cached,
                        dstOffset = androidx.compose.ui.unit.IntOffset(offX.toInt(), offY.toInt()),
                        dstSize = androidx.compose.ui.unit.IntSize(drawnW.toInt(), drawnH.toInt()),
                    )
                }
            } else {
                Text(
                    "(no still image — pass -PdesktopHarnessImage=<path>)",
                    color = Color(0xFFCCCCCC),
                    fontSize = 14.sp,
                    modifier = Modifier.align(Alignment.Center),
                )
            }
        }
    }

    private fun bufferedImageToImageBitmap(img: BufferedImage): ImageBitmap {
        val baos = ByteArrayOutputStream()
        ImageIO.write(img, "png", baos)
        val bytes = baos.toByteArray()
        return SkiaImage.makeFromEncoded(bytes).toComposeImageBitmap()
    }
}
