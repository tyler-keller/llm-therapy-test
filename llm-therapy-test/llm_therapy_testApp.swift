//
//  llm_therapy_testApp.swift
//  llm-therapy-test
//
//  Created by Tyler Keller on 4/25/24.
//

import SwiftUI

@main
struct llm_therapy_testApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
    }
}
