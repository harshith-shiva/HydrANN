package com.hydrann.ui;

import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.io.File;
import java.util.List;

/** The few knobs the app exposes. */
final class SettingsView extends VBox {

    SettingsView(AppContext ctx) {
        getStyleClass().add("page");
        setSpacing(16);

        HBox title = new HBox(16, PixelKit.sprite("mascot_44"), PixelKit.label("Settings", "page-title"));
        title.setAlignment(Pos.CENTER_LEFT);

        PixelDropdown count = new PixelDropdown("", List.of("5", "10", "20"), String.valueOf(ctx.resultCount.get()));
        count.setOnChange(v -> ctx.resultCount.set(Integer.parseInt(v)));

        PixelDropdown mode = new PixelDropdown("", List.of("Semantic", "Exact scan"), ctx.exactMode.get() ? "Exact scan" : "Semantic");
        mode.setOnChange(v -> ctx.exactMode.set(v.startsWith("Exact")));
        ctx.exactMode.addListener((o, a, b) -> mode.setValue(b ? "Exact scan" : "Semantic"));

        Button openFolder = new Button("Open folder");
        openFolder.getStyleClass().add("small-button");
        openFolder.setOnAction(e -> {
            SearchBackend b = ctx.backend;
            if (b == null || ctx.host == null) return;
            File f = new File(b.libraryFolder());
            if (f.isDirectory()) ctx.host.showDocument(f.toURI().toString());
            else ctx.toast.accept("Folder not found: " + f);
        });
        Button scan = new Button("Scan for new PDFs");
        scan.getStyleClass().add("small-button");
        scan.setOnAction(e -> ctx.rescan.run());

        Label about = PixelKit.label(
                "HydrANN runs entirely on this computer. Passages are embedded with MiniLM, stored on disk, "
                        + "and searched by whichever index the Hydra controller currently judges best.", "lab-explain");
        about.setWrapText(true);

        VBox panel = new VBox(14,
                row("Results per search", "How many passages to show.", count),
                row("Search mode", "Semantic uses the live index; Exact scan checks every passage.", mode),
                row("Library folder", "PDFs placed here are picked up by a scan.", new HBox(10, openFolder, scan)),
                about);
        panel.getStyleClass().add("pixel-panel");

        getChildren().addAll(title, panel);
        VBox.setVgrow(panel, Priority.NEVER);
    }

    private static HBox row(String name, String hint, javafx.scene.Node control) {
        VBox text = new VBox(2, PixelKit.label(name, "card-title"), PixelKit.label(hint, "card-meta"));
        HBox.setHgrow(text, Priority.ALWAYS);
        HBox r = new HBox(14, text, control);
        r.setAlignment(Pos.CENTER_LEFT);
        r.getStyleClass().add("stat-row");
        return r;
    }
}
