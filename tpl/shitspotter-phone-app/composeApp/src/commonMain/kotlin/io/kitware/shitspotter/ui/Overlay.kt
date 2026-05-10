package io.kitware.shitspotter.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.translate
import io.kitware.shitspotter.core.Detection

/**
 * Scaling mode for [DetectionOverlay]. Mirrors the meaningful subset of
 * Android's PreviewView scale types so the overlay can stay aligned with
 * the camera preview.
 *
 * - [FIT_CENTER] picks the smallest scale that fits the frame entirely
 *   inside the canvas (letterbox). Detection boxes always fall inside
 *   visible canvas space.
 * - [FILL_CENTER] picks the largest scale that fills the canvas (crop
 *   excess). Detection boxes near the cropped edges may render
 *   partially or fully off-canvas — but the overlay aligns with the
 *   camera preview, which is what the user actually sees.
 */
enum class OverlayScaleMode { FIT_CENTER, FILL_CENTER }

/**
 * Draws bounding boxes over the parent layout's pixel space, optionally
 * matching a PreviewView FILL_CENTER scale type so the boxes line up
 * with the on-screen camera image rather than a letterboxed overlay.
 *
 * [frameWidth] / [frameHeight] are the source dimensions of the frame the
 * detections were produced in.
 */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier,
    boxColor: Color = Color(0xFFFF4D4D),
    strokeWidthPx: Float = 4f,
    scaleMode: OverlayScaleMode = OverlayScaleMode.FILL_CENTER,
) {
    Box(modifier = modifier) {
        if (frameWidth <= 0 || frameHeight <= 0 || detections.isEmpty()) return@Box
        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleX = size.width / frameWidth.toFloat()
            val scaleY = size.height / frameHeight.toFloat()
            val scale = when (scaleMode) {
                OverlayScaleMode.FIT_CENTER -> minOf(scaleX, scaleY)
                OverlayScaleMode.FILL_CENTER -> maxOf(scaleX, scaleY)
            }
            val drawnW = frameWidth * scale
            val drawnH = frameHeight * scale
            val offX = (size.width - drawnW) / 2f
            val offY = (size.height - drawnH) / 2f
            translate(offX, offY) {
                detections.forEach { d ->
                    val l = d.box.left * scale
                    val t = d.box.top * scale
                    val w = d.box.width * scale
                    val h = d.box.height * scale
                    drawRect(
                        color = boxColor,
                        topLeft = Offset(l, t),
                        size = Size(w, h),
                        style = Stroke(width = strokeWidthPx),
                    )
                }
            }
        }
    }
}

@Composable
fun PlaceholderPreviewBackground(modifier: Modifier = Modifier) {
    Box(modifier = modifier.background(Color(0xFF101010)))
}
