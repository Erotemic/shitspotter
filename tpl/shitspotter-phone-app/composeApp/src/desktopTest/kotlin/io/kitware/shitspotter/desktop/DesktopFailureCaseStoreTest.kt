package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.BoundingBox
import io.kitware.shitspotter.core.Detection
import io.kitware.shitspotter.core.FailureCaseMetadata
import io.kitware.shitspotter.core.FailureType
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DesktopFailureCaseStoreTest {

    private fun sampleMetadata(ts: String = "2026-05-10T17-00-00Z"): FailureCaseMetadata =
        FailureCaseMetadata(
            timestamp = ts,
            deviceModel = "test", osVersion = "test", appCommit = "deadbeef",
            modelId = "stub-fake-detector", modelHash = null,
            runtimeBackend = "stub-1.0", delegate = null,
            inputWidth = 1, inputHeight = 1,
            scoreThreshold = 0.25f, iouThreshold = 0.45f,
            latencyMs = 0.0, fpsRecent = 0.0,
            failureType = FailureType.UNCERTAIN, userNote = null,
            detections = listOf(
                Detection(BoundingBox(0f, 0f, 1f, 1f), score = 0.5f, classId = 0, className = "poop"),
            ),
        )

    @Test
    fun save_writes_metadata_and_detections_files() {
        val root = createTempDirectory("ssp-store").toFile()
        try {
            val store = DesktopFailureCaseStore(root)
            val path = store.save(ByteArray(0), sampleMetadata())
            val dir = File(path)
            assertTrue(dir.isDirectory)
            assertTrue(File(dir, "metadata.json").isFile)
            assertTrue(File(dir, "detections.json").isFile)
            // No image bytes were given, so the file should not exist.
            assertTrue(!File(dir, "image.jpg").exists())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun save_writes_image_when_bytes_provided() {
        val root = createTempDirectory("ssp-store").toFile()
        try {
            val store = DesktopFailureCaseStore(root)
            val bytes = ByteArray(64) { 0x42 }
            val path = store.save(bytes, sampleMetadata())
            val img = File(File(path), "image.jpg")
            assertTrue(img.isFile)
            assertEquals(64L, img.length())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun save_writes_user_note_when_provided() {
        val root = createTempDirectory("ssp-store").toFile()
        try {
            val store = DesktopFailureCaseStore(root)
            val md = sampleMetadata().copy(userNote = "looks like grass")
            val path = store.save(ByteArray(0), md)
            val note = File(File(path), "user_note.txt")
            assertTrue(note.isFile)
            assertEquals("looks like grass", note.readText())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun save_creates_unique_dirs_for_each_call() {
        val root = createTempDirectory("ssp-store").toFile()
        try {
            val store = DesktopFailureCaseStore(root)
            val a = store.save(ByteArray(0), sampleMetadata("2026-05-10T17-00-00Z"))
            val b = store.save(ByteArray(0), sampleMetadata("2026-05-10T17-00-01Z"))
            assertTrue(a != b)
        } finally {
            root.deleteRecursively()
        }
    }
}
