package io.kitware.shitspotter.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.Fmt
import io.kitware.shitspotter.core.FrameTelemetry

@Composable
fun TelemetryHud(
    telemetry: FrameTelemetry?,
    detectionCount: Int,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .background(Color(0x88000000), RoundedCornerShape(8.dp))
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        if (telemetry == null) {
            Text("waiting for first frame…", color = Color.White, fontSize = 12.sp)
            return@Column
        }
        Row {
            Text("FPS ", color = Color.White, fontSize = 14.sp)
            Text(Fmt.ms(telemetry.fpsRecent), color = Color(0xFFB7E3FF), fontSize = 14.sp)
            Spacer(Modifier.width(12.dp))
            Text("dets ", color = Color.White, fontSize = 14.sp)
            Text("$detectionCount", color = Color(0xFFB7E3FF), fontSize = 14.sp)
        }
        Row {
            Text("inf ", color = Color.White, fontSize = 12.sp)
            Text("${Fmt.ms(telemetry.inferenceMs)} ms", color = Color(0xFFB7E3FF), fontSize = 12.sp)
            Spacer(Modifier.width(8.dp))
            Text("pre ", color = Color.White, fontSize = 12.sp)
            Text(Fmt.ms(telemetry.preprocessMs), color = Color(0xFFB7E3FF), fontSize = 12.sp)
            Spacer(Modifier.width(8.dp))
            Text("post ", color = Color.White, fontSize = 12.sp)
            Text(Fmt.ms(telemetry.postprocessMs), color = Color(0xFFB7E3FF), fontSize = 12.sp)
        }
        Text(
            "${telemetry.runtimeBackend} | ${telemetry.delegate ?: "cpu"} | ${telemetry.inputWidth}x${telemetry.inputHeight}",
            color = Color(0xFFCCCCCC),
            fontSize = 10.sp,
        )
        Text(
            "model ${telemetry.modelId}",
            color = Color(0xFFCCCCCC),
            fontSize = 10.sp,
        )
        Text(
            "sha=${telemetry.appCommit} ${BuildInfo.buildDate} | dropped ${telemetry.droppedFrames}",
            color = Color(0xFF888888),
            fontSize = 9.sp,
        )
    }
}

