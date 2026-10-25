package com.hydrann.ui;

import javafx.application.Application;

/**
 * Entry point. Kept separate from {@link HydraApp} so the app also starts from a plain classpath
 * (VS Code, "mvn exec:java") without the "JavaFX runtime components are missing" error.
 *
 * <pre>
 *   Launcher          real PDFs, real embeddings, live Hydra controller
 *   Launcher demo     built-in sample passages (no model or PDFs needed)
 * </pre>
 */
public final class Launcher {

    private Launcher() {}

    public static void main(String[] args) {
        Application.launch(HydraApp.class, args);
    }
}
