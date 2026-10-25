package com.hydrann.ui;

import javafx.application.HostServices;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

/** State and callbacks shared by every view. */
final class AppContext {

    final ExecutorService worker;          // searches, scans
    final ExecutorService statusWorker;    // 1 Hz status polling

    final IntegerProperty resultCount = new SimpleIntegerProperty(10);
    final BooleanProperty exactMode = new SimpleBooleanProperty(false);
    final StringProperty mascotLine = new SimpleStringProperty("Waking up...");

    volatile SearchBackend backend;
    HostServices host;
    Consumer<String> toast = s -> {};
    Runnable rescan = () -> {};

    AppContext(ExecutorService worker, ExecutorService statusWorker) {
        this.worker = worker;
        this.statusWorker = statusWorker;
    }
}
