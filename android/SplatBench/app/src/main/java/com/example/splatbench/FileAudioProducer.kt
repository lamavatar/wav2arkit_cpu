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
 * Decodes a file URI into 16kHz mono u8 PCM. Every [AppConfig.POLL_MS] writes
 * [AppConfig.fileTickBytes] (500ms) into the shared [AudioBuffer], then waits
 * 200ms before the next tick (2.5× realtime fill rate).
 */
class FileAudioProducer(
    private val context: Context,
    private val uri: Uri,
) : AudioProducer {

    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private var session: PlaybackSession? = null

    override fun isRunning(): Boolean = running.get()

    override fun start(buffer: AudioBuffer, session: PlaybackSession) {
        if (!running.compareAndSet(false, true)) return
        this.session = session
        val t = Thread({
            try {
                tickLoop(buffer, session)
            } catch (e: Exception) {
                Log.w(TAG, "decode failed: ${e.message}")
            } finally {
                running.set(false)
            }
        }, "audio-decode-file")
        thread = t
        t.start()
    }

    override fun stop() {
        running.set(false)
        session?.requestStop()
        thread?.interrupt()
        thread = null
    }

    private fun tickLoop(buffer: AudioBuffer, session: PlaybackSession) {
        val decoder = FilePcmDecoder(context, uri)
        decoder.open()
        try {
            val tickBytes = AppConfig.fileTickBytes
            val pollTickBytes = AppConfig.pollTickBytes
            val scratch = ByteArray(tickBytes)
            while (running.get() && !session.isStopRequested()) {
                val tickStart = System.nanoTime()
                val n = decoder.readU8(scratch, tickBytes)
                if (n > 0) {
                    val written = buffer.writeBlocking(scratch, n) {
                        running.get() && !session.isStopRequested()
                    }
                    if (written < n && buffer.freeBytes() == 0) {
                        Log.w(TAG, "audio buffer full before EOF — truncating at ${buffer.durationMs()}ms")
                        buffer.markEndOfStream()
                        break
                    }
                }
                if (decoder.isEof()) {
                    val pad = AppConfig.chunkBytes - (buffer.availableBytes() % AppConfig.chunkBytes)
                    if (pad in 1 until AppConfig.chunkBytes) {
                        buffer.writeSilence(pad)
                    }
                    buffer.writeSilence(pollTickBytes)
                    buffer.markEndOfStream()
                    break
                }
                sleepUntil(tickStart + AppConfig.POLL_MS * 1_000_000L)
            }
        } finally {
            decoder.close()
        }
    }

    private fun sleepUntil(deadlineNs: Long) {
        val rem = deadlineNs - System.nanoTime()
        if (rem > 0) {
            try {
                Thread.sleep(rem / 1_000_000L, (rem % 1_000_000L).toInt())
            } catch (_: InterruptedException) {
            }
        }
    }

    /** Incremental MediaCodec decoder that yields fixed-size u8 blocks. */
    private class FilePcmDecoder(
        private val context: Context,
        private val uri: Uri,
    ) {
        private var extractor: MediaExtractor? = null
        private var codec: MediaCodec? = null
        private var resampler: LinearResampler? = null
        private val staging = ArrayList<Byte>(8192)
        private var eof = false

        fun open() {
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

            this.extractor = extractor
            this.codec = codec
            this.resampler = LinearResampler(srcSr, AppConfig.AUDIO_SR)
        }

        fun isEof(): Boolean = eof && staging.isEmpty()

        /**
         * Fill [out] with up to [maxBytes] u8 samples. Returns bytes written
         * (may be less than max at EOF).
         */
        fun readU8(out: ByteArray, maxBytes: Int): Int {
            while (staging.size < maxBytes && !eof) {
                if (!pumpCodec()) eof = true
            }
            val n = minOf(maxBytes, staging.size)
            for (i in 0 until n) out[i] = staging[i]
            repeat(n) { staging.removeAt(0) }
            return n
        }

        fun close() {
            try { codec?.stop() } catch (_: Exception) {}
            codec?.release()
            extractor?.release()
            codec = null
            extractor = null
            staging.clear()
        }

        private fun pumpCodec(): Boolean {
            val extractor = extractor ?: return false
            val codec = codec ?: return false
            val resampler = resampler ?: return false
            val bufInfo = MediaCodec.BufferInfo()
            val timeoutUs = 10_000L
            var progressed = false

            val inIdx = codec.dequeueInputBuffer(timeoutUs)
            if (inIdx >= 0) {
                val inBuf = codec.getInputBuffer(inIdx)!!
                val sampleSize = extractor.readSampleData(inBuf, 0)
                if (sampleSize < 0) {
                    codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                } else {
                    codec.queueInputBuffer(inIdx, 0, sampleSize, extractor.sampleTime, 0)
                    extractor.advance()
                }
                progressed = true
            }

            val outIdx = codec.dequeueOutputBuffer(bufInfo, timeoutUs)
            if (outIdx >= 0) {
                if (bufInfo.size > 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.position(bufInfo.offset)
                    outBuf.limit(bufInfo.offset + bufInfo.size)
                    val mono16 = downmixToMono16(outBuf, channelCount(extractor))
                    val res = resampler.process(mono16)
                    for (s in res) staging.add(AudioPcmConverter.i16ToU8(s))
                }
                codec.releaseOutputBuffer(outIdx, false)
                if (bufInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    eof = true
                }
                progressed = true
            }
            return progressed
        }

        private fun channelCount(extractor: MediaExtractor): Int {
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                if (fmt.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                    return if (fmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
                        fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1
                }
            }
            return 1
        }

        private fun selectAudioTrack(extractor: MediaExtractor): Int {
            for (i in 0 until extractor.trackCount) {
                val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) return i
            }
            return -1
        }

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

        private class LinearResampler(private val srcSr: Int, private val dstSr: Int) {
            private val ratio = srcSr.toDouble() / dstSr.toDouble()
            private var pos = 0.0
            private var last: Short = 0
            private var primed = false

            fun process(input: ShortArray): ShortArray {
                if (input.isEmpty()) return ShortArray(0)
                if (srcSr == dstSr) return input

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
                pos -= (ext.size - 1)
                if (pos < 0.0) pos = 0.0
                last = input[input.size - 1]
                primed = true
                return out.toShortArray()
            }
        }
    }

    companion object {
        private const val TAG = "FileAudioProducer"
    }
}
