package com.example.splatbench

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Streams an audio file (URI) into the shared ring buffer the same way the mic
 * does: decode → mono → 16kHz resample → u8 → [AudioRingBuffer.write]. The full
 * file is never held in memory; the producer stays ahead of inference/playback
 * and signals [AudioRingBuffer.markEndOfStream] at EOF.
 */
class FileAudioProducer(
    private val context: Context,
    private val uri: Uri,
) : AudioProducer {

    private val running = AtomicBoolean(false)
    private var thread: Thread? = null

    override fun isRunning(): Boolean = running.get()

    override fun start(ring: AudioRingBuffer) {
        if (!running.compareAndSet(false, true)) return
        val t = Thread({
            try {
                decodeLoop(ring)
            } catch (e: Exception) {
                Log.w(TAG, "decode failed: ${e.message}")
            } finally {
                ring.markEndOfStream()
                running.set(false)
            }
        }, "audio-decode-file")
        thread = t
        t.start()
    }

    override fun stop() {
        running.set(false)
        thread?.interrupt()
        thread = null
    }

    private fun decodeLoop(ring: AudioRingBuffer) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        val trackIndex = selectAudioTrack(extractor)
        require(trackIndex >= 0) { "no audio track" }
        extractor.selectTrack(trackIndex)
        val format = extractor.getTrackFormat(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: error("no mime")
        val srcSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val srcCh = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
            format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val resampler = LinearResampler(srcSr, AppConfig.AUDIO_SR)
        val bufInfo = MediaCodec.BufferInfo()
        var sawInputEos = false
        var sawOutputEos = false
        val timeoutUs = 10_000L

        // Reusable u8 output staging.
        var u8 = ByteArray(8192)

        while (running.get() && !sawOutputEos) {
            if (!sawInputEos) {
                val inIdx = codec.dequeueInputBuffer(timeoutUs)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx)!!
                    val sampleSize = extractor.readSampleData(inBuf, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEos = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outIdx = codec.dequeueOutputBuffer(bufInfo, timeoutUs)
            if (outIdx >= 0) {
                if (bufInfo.size > 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.position(bufInfo.offset)
                    outBuf.limit(bufInfo.offset + bufInfo.size)
                    val mono16 = downmixToMono16(outBuf, srcCh)
                    val res = resampler.process(mono16)
                    if (res.isNotEmpty()) {
                        if (u8.size < res.size) u8 = ByteArray(res.size)
                        for (i in res.indices) u8[i] = AudioPcmConverter.i16ToU8(res[i])
                        writeThrottled(ring, u8, res.size)
                    }
                }
                codec.releaseOutputBuffer(outIdx, false)
                if (bufInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    sawOutputEos = true
                }
            }
        }

        try { codec.stop() } catch (_: Exception) {}
        codec.release()
        extractor.release()
    }

    /**
     * Keep the producer ahead of the consumer without overrunning the 10s ring:
     * if the ring is nearly full, wait for the inference thread to drain it.
     */
    private fun writeThrottled(ring: AudioRingBuffer, u8: ByteArray, len: Int) {
        var offset = 0
        while (offset < len && running.get()) {
            val free = ring.capacityBytes - ring.availableBytes()
            if (free <= 0) {
                try { Thread.sleep(5) } catch (_: InterruptedException) { return }
                continue
            }
            val n = minOf(free, len - offset)
            val block = if (offset == 0 && n == len) u8 else u8.copyOfRange(offset, offset + n)
            ring.write(block, n)
            offset += n
        }
    }

    private fun selectAudioTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return -1
    }

    /** Interleaved 16-bit PCM → mono 16-bit (average channels). */
    private fun downmixToMono16(buf: ByteBuffer, channels: Int): ShortArray {
        val sb = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val total = sb.remaining()
        if (channels <= 1) {
            val out = ShortArray(total)
            sb.get(out)
            return out
        }
        val frames = total / channels
        val out = ShortArray(frames)
        val tmp = ShortArray(total)
        sb.get(tmp)
        var si = 0
        for (f in 0 until frames) {
            var acc = 0
            for (c in 0 until channels) acc += tmp[si++].toInt()
            out[f] = (acc / channels).toShort()
        }
        return out
    }

    /** Stateful linear resampler (keeps fractional phase + last sample across blocks). */
    private class LinearResampler(private val srcSr: Int, private val dstSr: Int) {
        private val ratio = srcSr.toDouble() / dstSr.toDouble()
        private var pos = 0.0
        private var last: Short = 0
        private var primed = false

        fun process(input: ShortArray): ShortArray {
            if (input.isEmpty()) return ShortArray(0)
            if (srcSr == dstSr) return input

            // Prepend the previous block's last sample so interpolation is
            // continuous across block boundaries.
            val ext: ShortArray
            if (primed) {
                ext = ShortArray(input.size + 1)
                ext[0] = last
                System.arraycopy(input, 0, ext, 1, input.size)
            } else {
                ext = input
            }

            val out = ArrayList<Short>((input.size / ratio).toInt() + 2)
            while (true) {
                val i = pos.toInt()
                if (i + 1 >= ext.size) break
                val frac = pos - i
                val s0 = ext[i].toDouble()
                val s1 = ext[i + 1].toDouble()
                val v = s0 + (s1 - s0) * frac
                out.add(v.toInt().coerceIn(-32768, 32767).toShort())
                pos += ratio
            }
            // Carry phase relative to next block's prepended last sample (ext[0]).
            pos -= (ext.size - 1)
            if (pos < 0.0) pos = 0.0
            last = input[input.size - 1]
            primed = true
            return out.toShortArray()
        }
    }

    companion object {
        private const val TAG = "FileAudioProducer"
    }
}
