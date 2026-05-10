package io.kitware.shitspotter.desktop

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class CompareCliArgsTest {

    @Test
    fun argValue_extracts_simple_kv() {
        val args = arrayOf("--image=foo.jpg", "--runs=5")
        assertEquals("foo.jpg", CompareCli.argValue(args, "--image"))
        assertEquals("5", CompareCli.argValue(args, "--runs"))
    }

    @Test
    fun argValue_returns_null_for_missing() {
        assertNull(CompareCli.argValue(emptyArray(), "--image"))
        assertNull(CompareCli.argValue(arrayOf("--runs=5"), "--image"))
    }

    @Test
    fun argValue_returns_first_match() {
        val args = arrayOf("--image=a.jpg", "--image=b.jpg")
        assertEquals("a.jpg", CompareCli.argValue(args, "--image"))
    }

    @Test
    fun argValues_collects_repeats_in_order() {
        val args = arrayOf("--model=a.onnx", "--runs=5", "--model=b.onnx", "--model=c.onnx")
        assertEquals(listOf("a.onnx", "b.onnx", "c.onnx"), CompareCli.argValues(args, "--model"))
    }

    @Test
    fun argValues_empty_when_none() {
        assertEquals(emptyList(), CompareCli.argValues(arrayOf("--runs=5"), "--model"))
    }

    @Test
    fun guessModelIdFromPath_finds_yolox_nano() {
        val id = CompareCli.guessModelIdFromPath("yolox_nano_poop_cropped_only_best.onnx")
        assertEquals("yolox-nano-poop-cropped-v1", id)
    }

    @Test
    fun guessModelIdFromPath_returns_null_for_unknown() {
        assertNull(CompareCli.guessModelIdFromPath("totally_made_up.onnx"))
    }

    @Test
    fun guessModelIdFromPath_is_case_insensitive() {
        val id = CompareCli.guessModelIdFromPath("YOLOX_NANO_POOP_CROPPED_ONLY_BEST.onnx")
        assertEquals("yolox-nano-poop-cropped-v1", id)
    }
}
