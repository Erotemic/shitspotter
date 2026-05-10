package io.kitware.shitspotter.core

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SerializationTest {

    @Test
    fun failure_case_round_trip() {
        val md = FailureCaseMetadata(
            timestamp = "2026-05-10T17:00:00Z",
            deviceModel = "test-device",
            osVersion = "test-os",
            appCommit = "deadbeef",
            modelId = "stub-fake-detector",
            modelHash = "abc",
            runtimeBackend = "stub-1.0",
            delegate = null,
            inputWidth = 640,
            inputHeight = 640,
            scoreThreshold = 0.25f,
            iouThreshold = 0.45f,
            latencyMs = 12.34,
            fpsRecent = 30.0,
            failureType = FailureType.FALSE_POSITIVE,
            userNote = "looks like a leaf",
            detections = listOf(
                Detection(BoundingBox(10f, 20f, 30f, 40f), score = 0.8f, classId = 0, className = "poop"),
            ),
        )
        val json = FailureCaseSerialization.json
        val s = json.encodeToString(md)
        // Pretty-printed; we expect newlines and the failure type by name.
        assertTrue(s.contains("FALSE_POSITIVE"))
        assertTrue(s.contains("\"score\": 0.8"))

        val back = json.decodeFromString<FailureCaseMetadata>(s)
        assertEquals(md, back)
    }

    @Test
    fun model_spec_round_trip() {
        val s = FailureCaseSerialization.json.encodeToString(ModelSpec.YOLOX_NANO_POOP)
        val back = FailureCaseSerialization.json.decodeFromString<ModelSpec>(s)
        assertEquals(ModelSpec.YOLOX_NANO_POOP, back)
    }

    @Test
    fun backend_comparison_report_round_trip() {
        val rep = BackendComparisonReport(
            timestamp = "2026-05-10T17:00:00Z",
            deviceModel = "test",
            osVersion = "test",
            rows = listOf(
                BackendRunRow(
                    frameWidth = 640, frameHeight = 480,
                    backendName = "stub-1.0", delegate = null,
                    modelId = "stub", inputWidth = 416, inputHeight = 416,
                    preprocessMs = 1.0, inferenceMs = 2.0, postprocessMs = 3.0,
                    detectionCount = 0, topScore = 0.0f,
                ),
            ),
        )
        val s = FailureCaseSerialization.json.encodeToString(rep)
        val back = FailureCaseSerialization.json.decodeFromString<BackendComparisonReport>(s)
        assertEquals(rep, back)
    }
}
