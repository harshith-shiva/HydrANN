package com.hydrann.ui;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.io.File;
import java.util.List;

/** Every PDF the engine has indexed, with how many passages each contributes. */
final class LibraryView extends VBox {

    private final AppContext ctx;
    private final VBox rows = new VBox(8);
    private final Label summary = PixelKit.label("", "muted");

    LibraryView(AppContext ctx) {
        this.ctx = ctx;
        getStyleClass().add("page");
        setSpacing(16);

        HBox title = new HBox(16, PixelKit.sprite("mascot_44"), PixelKit.label("Library", "page-title"));
        title.setAlignment(Pos.CENTER_LEFT);

        Button scan = new Button("Scan for new PDFs");
        scan.getStyleClass().add("small-button");
        scan.setOnAction(e -> ctx.rescan.run());
        HBox top = new HBox(14, summary, PixelKit.hGrow(), scan);
        top.setAlignment(Pos.CENTER_LEFT);

        rows.setPadding(new Insets(2, 10, 2, 2));
        ScrollPane sp = new ScrollPane(rows);
        sp.setFitToWidth(true);
        sp.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        sp.getStyleClass().add("pixel-scroll");
        VBox.setVgrow(sp, Priority.ALWAYS);

        VBox panel = new VBox(12, top, sp);
        panel.getStyleClass().add("pixel-panel");
        VBox.setVgrow(panel, Priority.ALWAYS);

        getChildren().addAll(title, panel);
    }

    void update(List<SearchBackend.DocInfo> docs) {
        int chunks = 0;
        rows.getChildren().clear();
        for (SearchBackend.DocInfo d : docs) {
            chunks += d.chunks();
            Label name = PixelKit.label(d.fileName(), "card-title");
            Label path = PixelKit.label(d.path(), "card-meta");
            VBox text = new VBox(2, name, path);
            HBox.setHgrow(text, Priority.ALWAYS);
            text.setMinWidth(0);
            Label counts = PixelKit.label(d.chunks() + " passages  \u00B7  " + d.pages() + " pages", "muted");

            Button open = new Button("Open");
            open.getStyleClass().add("small-button");
            open.setOnAction(e -> {
                File f = new File(d.path());
                if (f.isFile() && ctx.host != null) ctx.host.showDocument(f.toURI().toString());
                else ctx.toast.accept("Could not find " + d.fileName() + " on disk.");
            });

            HBox row = new HBox(14, PixelKit.sprite("doc_card"), text, counts, open);
            row.setAlignment(Pos.CENTER_LEFT);
            row.getStyleClass().add("result-card");
            rows.getChildren().add(row);
        }
        if (docs.isEmpty()) {
            Label empty = PixelKit.label("No PDFs indexed yet. Drop files into " + ctx.backend.libraryFolder()
                    + " and press Scan.", "empty-text");
            empty.setWrapText(true);
            rows.getChildren().add(empty);
        }
        summary.setText(docs.size() + (docs.size() == 1 ? " document" : " documents") + "  \u00B7  " + chunks + " searchable passages");
    }
}
