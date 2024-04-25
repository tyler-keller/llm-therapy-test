//
//  ContentView.swift
//  llm-therapy-test
//
//  Created by Tyler Keller on 4/25/24.
//

import SwiftUI
import MLXLLM
import MLX
import MLXRandom
import Metal
import Tokenizers

struct ContentView: View {
    @State var llm = LLMEvaluator()
    @State var prompt = ""
    @Environment(DeviceStat.self) private var deviceStat

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                }
            }

            // show the model output
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        Text(llm.output)
                            .textSelection(.enabled)
                    }
                    .onChange(of: llm.output) { _, _ in
                        sp.scrollTo("bottom")
                    }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }

            HStack {
                TextField("prompt", text: $prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    .padding(.leading, 15)
                Button("generate", action: generate)
                    .disabled(llm.running)
                    .padding(.trailing, 15)
            }
            .padding(.bottom, 20)
        }
        .task {
            // pre-load the weights on launch to speed up the first generation
            _ = try? await llm.load()
        }
    }

    private func generate() {
        Task {
            await llm.generate(prompt: prompt)
        }
    }
}

#Preview {
    ContentView()
}

let customPhi3Config = ModelConfiguration(
    id: "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed"
) {
    prompt in
    "<s><|system|>You are an AI-powered counselor specialized in cognitive behavioral therapy. Your primary function is to engage users in self-reflection, challenge negative thought patters, and guide them towards adaptive behaviors. Your first step is always to get information about what's troubling the user. In subsequent responses, summarize the user's thoughts back to them, explain what they're feeling about the situation and why, then, ask the user a leading introspective question. Don't overpower the user with your own words, ask them leading questions and allow them to introspect. Be clear and concise, 2-3 sentences max. End your responses with <|endoftext|>.<|end|><|user|>\n\(prompt)<|end|>\n<|assistant|>\n"
}


@Observable
class LLMEvaluator {

    @MainActor
    var running = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// this controls which model loads -- phi4bit is one of the smaller ones so this will fit on
    /// more devices
    let modelConfiguration = customPhi3Config

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.6)
    let maxTokens = 240

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    enum LoadState {
        case idle
        case loaded(LLMModel, Tokenizers.Tokenizer)
    }

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> (LLMModel, Tokenizers.Tokenizer) {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let (model, tokenizer) = try await MLXLLM.load(configuration: modelConfiguration) {
                [modelConfiguration] progress in
                DispatchQueue.main.sync {
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(MLX.GPU.activeMemory / 1024 / 1024)M"
            loadState = .loaded(model, tokenizer)
            print(tokenizer.eosToken)
//            tokenizer.eosToken = "<|end|>"
            return (model, tokenizer)

        case .loaded(let model, let tokenizer):
            return (model, tokenizer)
        }
    }

    func generate(prompt: String) async {
        let canGenerate = await MainActor.run {
            if running {
                return false
            } else {
                running = true
                self.output = ""
                return true
            }
        }

        guard canGenerate else { return }

        do {
            let (model, tokenizer) = try await load()
            // augment the prompt as needed
            let prompt = modelConfiguration.prepare(prompt: prompt)
            let promptTokens = tokenizer.encode(text: prompt)

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let result = await MLXLLM.generate(
                promptTokens: promptTokens, parameters: generateParameters, model: model,
                tokenizer: tokenizer
            ) { tokens in
                // update the output -- this will make the view show the text as it generates
                if tokens.count % displayEveryNTokens == 0 {
                    let text = tokenizer.decode(tokens: tokens)
                    await MainActor.run {
                        self.output = text
                    }
                }

                if tokens.count >= maxTokens || tokenizer.decode(tokens: [tokens.last!]) == "<|end|>"{
                    return .stop
                } else {
                    return .more
                }
                
            }

            // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
            await MainActor.run {
                if result.output != self.output {
                    self.output = result.output
                }
                running = false
                self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"
            }

        } catch {
            await MainActor.run {
                running = false
                output = "Failed: \(error)"
            }
        }
    }
}
