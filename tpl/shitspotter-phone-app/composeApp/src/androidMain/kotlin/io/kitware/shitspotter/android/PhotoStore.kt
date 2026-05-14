package io.kitware.shitspotter.android

import android.location.Location
import android.os.Build
import androidx.exifinterface.media.ExifInterface
import io.kitware.shitspotter.core.CaptureLabel
import io.kitware.shitspotter.core.CaptureMetadata
import io.kitware.shitspotter.core.FailureCaseSerialization
import io.kitware.shitspotter.core.MetadataMode
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import java.io.File

class PhotoStore(private val dir: File) {
    init { dir.mkdirs() }

    fun newCaptureFile(timestamp: String): File {
        val safe = timestamp.replace(":", "-").replace(".", "-").replace("T", "_")
        return File(dir, "capture_${safe}.jpg")
    }

    fun addMetadataAndSave(
        jpegFile: File,
        metadata: CaptureMetadata,
        mode: MetadataMode,
        location: Location?,
    ): File {
        if (mode != MetadataMode.NONE) {
            val exif = ExifInterface(jpegFile.absolutePath)
            exif.setAttribute(ExifInterface.TAG_MAKE, Build.MANUFACTURER)
            exif.setAttribute(ExifInterface.TAG_MODEL, Build.MODEL)
            exif.setAttribute(ExifInterface.TAG_SOFTWARE, "ShitSpotter ${metadata.appCommit}")
            val ts = metadata.timestamp
            val date = ts.take(10).replace("-", ":")
            val time = if (ts.length >= 19) ts.substring(11, 19) else "00:00:00"
            exif.setAttribute(ExifInterface.TAG_DATETIME, "$date $time")
            exif.setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, "$date $time")
            if (mode == MetadataMode.FULL && location != null) {
                exif.setAttribute(ExifInterface.TAG_GPS_LATITUDE, decToDms(Math.abs(location.latitude)))
                exif.setAttribute(ExifInterface.TAG_GPS_LATITUDE_REF, if (location.latitude >= 0) "N" else "S")
                exif.setAttribute(ExifInterface.TAG_GPS_LONGITUDE, decToDms(Math.abs(location.longitude)))
                exif.setAttribute(ExifInterface.TAG_GPS_LONGITUDE_REF, if (location.longitude >= 0) "E" else "W")
                if (location.hasAltitude()) {
                    val alt = location.altitude
                    exif.setAttribute(ExifInterface.TAG_GPS_ALTITUDE, "${(Math.abs(alt) * 1000).toLong()}/1000")
                    exif.setAttribute(ExifInterface.TAG_GPS_ALTITUDE_REF, if (alt < 0) "1" else "0")
                }
            }
            exif.saveAttributes()
        }
        val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
        jsonFile.writeText(FailureCaseSerialization.json.encodeToString(metadata))
        return jpegFile
    }

    fun listAll(): List<Pair<File, CaptureMetadata?>> {
        val jpegFiles = dir.listFiles { f -> f.extension == "jpg" || f.extension == "jpeg" }
            ?: return emptyList()
        return jpegFiles.sortedByDescending { it.lastModified() }.map { jpegFile ->
            val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
            val meta = if (jsonFile.exists()) {
                try { FailureCaseSerialization.json.decodeFromString<CaptureMetadata>(jsonFile.readText()) }
                catch (_: Throwable) { null }
            } else null
            Pair(jpegFile, meta)
        }
    }

    fun updateLabel(jpegFile: File, newLabel: CaptureLabel, note: String?) {
        val jsonFile = File(jpegFile.parent, jpegFile.nameWithoutExtension + ".json")
        if (!jsonFile.exists()) return
        val existing = try {
            FailureCaseSerialization.json.decodeFromString<CaptureMetadata>(jsonFile.readText())
        } catch (_: Throwable) { return }
        val updated = existing.copy(label = newLabel, userNote = note ?: existing.userNote)
        jsonFile.writeText(FailureCaseSerialization.json.encodeToString(updated))
    }

    private fun decToDms(decimal: Double): String {
        val deg = decimal.toInt()
        val minFull = (decimal - deg) * 60.0
        val min = minFull.toInt()
        val secMicro = ((minFull - min) * 60.0 * 1_000_000).toLong()
        return "$deg/1,$min/1,$secMicro/1000000"
    }
}
