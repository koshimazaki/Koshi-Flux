/**
 * ComfyUI LTX-Audio Nodes - Web Extensions
 *
 * Custom JavaScript for audio node visualization and interaction.
 */

import { app } from "../../scripts/app.js";

// Register extension
app.registerExtension({
    name: "LTX-Audio",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add custom behavior for specific nodes

        if (nodeData.name === "AudioPreview") {
            // Custom preview node behavior
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                // Could add custom preview handling here
            };
        }

        if (nodeData.name === "BeatDetector") {
            // Visual feedback for beat detection
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                this.color = "#2a4858";
                this.bgcolor = "#1a2e38";
            };
        }

        // Style audio-related nodes with consistent colors
        const audioNodes = [
            "LoadAudio", "AudioEncoder", "AudioPreview", "ExtractAudioFeatures",
            "TranscribeAudio", "SpeechToPrompts", "TemporalPromptScheduler",
            "VoiceDrivenGenerator", "AudioParameterMapper", "AudioReactivePreset",
            "BeatDetector", "AudioToDeforumSchedule", "LTXAudioConditioner",
            "LTXAudioAdapterLoader", "LTXAudioLoRALoader", "CombineAudioVideo"
        ];

        if (audioNodes.includes(nodeData.name)) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                // Audio nodes get a teal/cyan accent
                this.color = "#00d4aa";
                this.bgcolor = "#1a1a2e";
            };
        }
    },

    async nodeCreated(node) {
        // Custom node creation handling
        const audioNodeNames = [
            "LoadAudio", "AudioEncoder", "AudioPreview", "ExtractAudioFeatures",
            "TranscribeAudio", "SpeechToPrompts", "TemporalPromptScheduler",
            "VoiceDrivenGenerator", "AudioParameterMapper", "AudioReactivePreset",
            "BeatDetector", "AudioToDeforumSchedule", "LTXAudioConditioner",
            "LTXAudioAdapterLoader", "LTXAudioLoRALoader", "CombineAudioVideo"
        ];

        if (audioNodeNames.includes(node.comfyClass)) {
            // Set consistent styling for audio nodes
            node.color = "#00d4aa";
            node.bgcolor = "#1a1a2e";
        }
    }
});

// Audio waveform widget (for future use)
class AudioWaveformWidget {
    constructor(node, inputName, inputData) {
        this.node = node;
        this.inputName = inputName;
        this.waveformData = null;
    }

    draw(ctx, node, width, y, height) {
        if (!this.waveformData) return;

        ctx.save();
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(0, y, width, height);

        ctx.strokeStyle = "#00d4aa";
        ctx.lineWidth = 1;
        ctx.beginPath();

        const samples = this.waveformData;
        const step = samples.length / width;

        for (let x = 0; x < width; x++) {
            const idx = Math.floor(x * step);
            const sample = samples[idx] || 0;
            const sampleY = y + height / 2 - sample * height / 2;

            if (x === 0) {
                ctx.moveTo(x, sampleY);
            } else {
                ctx.lineTo(x, sampleY);
            }
        }

        ctx.stroke();
        ctx.restore();
    }

    setWaveform(data) {
        this.waveformData = data;
        this.node.setDirtyCanvas(true);
    }
}

// Export for potential use by other extensions
window.LTXAudioWidgets = {
    AudioWaveformWidget
};
