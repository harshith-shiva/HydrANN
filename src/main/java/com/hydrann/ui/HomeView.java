package com.hydrann.ui;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.css.PseudoClass;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.layout.FlowPane;
import javafx.scene.text.TextFlow;

import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;

/** The search screen: banner, search bar, result cards, index status and the mascot. */
final class HomeView extends VBox {

    private static final PseudoClass SELECTED = PseudoClass.getPseudoClass("selected");
    private static final List<String> DEFAULT_TRIES = List.of(
            "how does malaria spread", "can birds fly", "what is natural selection",
            "why do objects fall", "how do plants make food");

    private final AppContext ctx;

    private final TextField input = new TextField();
    private final PixelDropdown modeDropdown =
            new PixelDropdown("", List.of("Semantic", "Exact scan"), "Semantic");
    private final Button searchButton = new Button("Search");
    private final FlowPane chips = new FlowPane(8, 8);

    private final Label matchesLabel = PixelKit.label("ask a question to begin", "muted");
    private final Label servedLabel = PixelKit.label("", "muted-small");
    private final PixelDropdown sortDropdown =
            new PixelDropdown("Sort by: ", List.of("Relevance", "Document", "Page"), "Relevance");
    private final VBox cardColumn = new VBox(10);
    private final ScrollPane scroll = new ScrollPane(cardColumn);
    private final VBox resultsPanel = new VBox(10);

    private final VBox readerPanel = new VBox(12);
    private final Label readerTitle = PixelKit.label("", "reader-title");
    private final Label readerMeta = PixelKit.label("", "muted");
    private final VBox readerBody = new VBox();
    private final ScrollPane readerScroll = new ScrollPane(readerBody);
    private SearchBackend.Hit readerHit;

    private final Label statDocs = PixelKit.label("-", "stat-value");
    private final Label statVectors = PixelKit.label("-", "stat-value");
    private final Label statType = PixelKit.label("-", "stat-value");
    private final Label statEmbedder = PixelKit.label("-", "stat-value");
    private final Label bubble = PixelKit.label("", "speech-bubble");

    private final List<Card> cards = new ArrayList<>();
    private int selected = -1;
    private List<SearchBackend.Hit> hits = List.of();
    private List<String> tokens = List.of();
    private final Deque<String> history = new ArrayDeque<>();
    private boolean phaseOverride;

    HomeView(AppContext ctx) {
        this.ctx = ctx;
        getStyleClass().add("home-view");

        VBox body = new VBox(14, buildSearchPanel(), buildColumns());
        body.setPadding(new Insets(0, 20, 18, 20));
        VBox.setVgrow(body.getChildren().get(1), Priority.ALWAYS);
        VBox.setVgrow(body, Priority.ALWAYS);
        getChildren().addAll(buildBanner(), body);

        renderChips();
        showEmpty("Type a question above and press Enter.");
        bubble.textProperty().bind(ctx.mascotLine);
        addEventFilter(KeyEvent.KEY_PRESSED, this::onKey);
    }

    void focusSearch() {
        input.requestFocus();
    }

    // ------------------------------------------------------------------ banner

    private Node buildBanner() {
        ImageView mascot = PixelKit.sprite("mascot_120");
        Label title = PixelKit.label("HydrANN", "brand-title");
        Label tagline = PixelKit.label("Your knowledge, rooted locally.", "brand-tagline");
        VBox titles = new VBox(10, title, tagline);
        titles.setAlignment(Pos.CENTER_LEFT);
        HBox left = new HBox(22, mascot, titles);
        left.setAlignment(Pos.BOTTOM_LEFT);
        left.setPadding(new Insets(0, 0, 16, 30));

        HBox heart = new HBox(6, PixelKit.label("Your data stays here.", "badge-text"), PixelKit.sprite("icon_heart"));
        heart.setAlignment(Pos.CENTER_LEFT);
        VBox badgeText = new VBox(2,
                PixelKit.label("LOCAL ONLY", "badge-title"),
                PixelKit.label("No external APIs.", "badge-text"),
                heart);
        HBox badge = new HBox(14, PixelKit.sprite("icon_leaf"), badgeText);
        badge.setAlignment(Pos.CENTER_LEFT);
        badge.getStyleClass().addAll("pixel-panel", "badge-panel");
        badge.setMaxHeight(Region.USE_PREF_SIZE);

        BorderPane content = new BorderPane();
        content.setLeft(left);
        content.setRight(badge);
        BorderPane.setAlignment(left, Pos.BOTTOM_LEFT);
        BorderPane.setAlignment(badge, Pos.TOP_RIGHT);
        BorderPane.setMargin(badge, new Insets(16, 22, 0, 0));

        Pane sky = new Pane();
        sky.setMouseTransparent(true);
        ImageView cloud1 = PixelKit.sprite("cloud_big");
        ImageView cloud2 = PixelKit.sprite("cloud_small");
        ImageView cloud3 = PixelKit.sprite("cloud_small");
        cloud1.layoutXProperty().bind(sky.widthProperty().multiply(0.50));
        cloud1.setLayoutY(18);
        cloud2.layoutXProperty().bind(sky.widthProperty().multiply(0.72));
        cloud2.setLayoutY(10);
        cloud3.layoutXProperty().bind(sky.widthProperty().multiply(0.38));
        cloud3.setLayoutY(88);
        sky.getChildren().addAll(cloud1, cloud2, cloud3);

        Region grass = new Region();
        grass.getStyleClass().add("grass-strip");
        PixelKit.fixHeight(grass, 27);

        Pane ground = new Pane();
        ground.setMouseTransparent(true);
        String[] names = {"flower", "sprout_small", "flower_orange", "flower", "sprout_small", "flower"};
        double[] xs = {0.43, 0.56, 0.66, 0.79, 0.88, 0.97};
        for (int i = 0; i < names.length; i++) {
            ImageView v = PixelKit.sprite(names[i]);
            v.layoutXProperty().bind(ground.widthProperty().multiply(xs[i]));
            v.layoutYProperty().bind(ground.heightProperty().subtract(24 + v.getImage().getHeight() - 6));
            ground.getChildren().add(v);
        }

        StackPane banner = new StackPane(sky, grass, ground, content);
        StackPane.setAlignment(grass, Pos.BOTTOM_CENTER);
        banner.setMinHeight(150);
        banner.setPrefHeight(150);
        banner.setMaxHeight(150);
        banner.getStyleClass().add("banner");
        return banner;
    }

    // ------------------------------------------------------------------ search panel

    private Node buildSearchPanel() {
        input.setPromptText("Search your notes, PDFs and ideas...");
        input.getStyleClass().add("search-input");
        HBox.setHgrow(input, Priority.ALWAYS);
        input.setOnAction(e -> runSearch(input.getText()));

        Region divider = new Region();
        divider.getStyleClass().add("v-divider");
        modeDropdown.setOnChange(v -> ctx.exactMode.set(v.startsWith("Exact")));
        ctx.exactMode.addListener((o, a, b) -> modeDropdown.setValue(b ? "Exact scan" : "Semantic"));

        HBox field = new HBox(12, PixelKit.sprite("icon_search"), input, divider, modeDropdown);
        input.focusedProperty().addListener((o, was, now) -> {
            if (now) field.getStyleClass().add("focus");
            else field.getStyleClass().remove("focus");
        });
        field.setAlignment(Pos.CENTER_LEFT);
        field.getStyleClass().add("pixel-input");
        HBox.setHgrow(field, Priority.ALWAYS);

        searchButton.getStyleClass().add("primary-button");
        searchButton.setGraphic(PixelKit.sprite("icon_search_light"));
        searchButton.setGraphicTextGap(12);
        searchButton.setOnAction(e -> runSearch(input.getText()));

        HBox row = new HBox(14, field, searchButton);
        row.setAlignment(Pos.CENTER);

        HBox tryRow = new HBox(12, PixelKit.label("Try:", "try-label"), chips);
        tryRow.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(chips, Priority.ALWAYS);

        VBox panel = new VBox(12, row, tryRow);
        panel.getStyleClass().add("pixel-panel");
        return panel;
    }

    private void renderChips() {
        chips.getChildren().clear();
        LinkedHashSet<String> picks = new LinkedHashSet<>(history);
        for (String d : DEFAULT_TRIES) {
            if (picks.size() >= 4) break;
            picks.add(d);
        }
        for (String q : picks) {
            Button b = new Button(q.length() > 26 ? q.substring(0, 24) + "..." : q);
            b.getStyleClass().add("chip");
            b.setOnAction(e -> runSearch(q));
            chips.getChildren().add(b);
        }
    }

    // ------------------------------------------------------------------ columns

    private Node buildColumns() {
        Node results = buildResultsArea();
        Node right = buildRightColumn();
        HBox box = new HBox(16, results, right);
        HBox.setHgrow(results, Priority.ALWAYS);
        return box;
    }

    private Node buildResultsArea() {
        Region dot = new Region();
        dot.getStyleClass().add("dot");
        HBox header = new HBox(10, PixelKit.sprite("icon_doc"), PixelKit.label("Search Results", "panel-heading"),
                dot, matchesLabel, PixelKit.hGrow(), sortDropdown);
        header.setAlignment(Pos.CENTER_LEFT);
        sortDropdown.setOnChange(v -> renderHits());

        scroll.setFitToWidth(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.getStyleClass().add("pixel-scroll");
        scroll.setFocusTraversable(true);
        cardColumn.setPadding(new Insets(2, 10, 2, 2));
        VBox.setVgrow(scroll, Priority.ALWAYS);

        HBox hints = new HBox(8,
                PixelKit.sprite("icon_up"), PixelKit.sprite("icon_down"), PixelKit.label("to navigate", "hint"),
                PixelKit.label("|", "hint"),
                PixelKit.sprite("icon_enter"), PixelKit.label("Enter to open", "hint"),
                PixelKit.label("|", "hint"),
                PixelKit.label("Esc to go back", "hint"),
                PixelKit.hGrow(), servedLabel);
        hints.setAlignment(Pos.CENTER_LEFT);

        resultsPanel.getChildren().addAll(header, scroll, hints);
        resultsPanel.getStyleClass().add("pixel-panel");

        // reader (opens over the results with Enter or a double click)
        readerTitle.setWrapText(true);
        Button back = new Button("Back");
        back.getStyleClass().add("small-button");
        back.setOnAction(e -> closeReader());
        Button openPdf = new Button("Open PDF");
        openPdf.getStyleClass().add("small-button");
        openPdf.setOnAction(e -> openPdf());
        HBox bar = new HBox(10, back, PixelKit.hGrow(), openPdf);
        bar.setAlignment(Pos.CENTER_LEFT);

        readerBody.setPadding(new Insets(4, 12, 4, 4));
        readerScroll.setFitToWidth(true);
        readerScroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        readerScroll.getStyleClass().add("pixel-scroll");
        VBox.setVgrow(readerScroll, Priority.ALWAYS);
        readerPanel.getChildren().addAll(bar, readerTitle, readerMeta, readerScroll);
        readerPanel.getStyleClass().add("pixel-panel");
        readerPanel.setVisible(false);

        return new StackPane(resultsPanel, readerPanel);
    }

    private Node buildRightColumn() {
        HBox head = new HBox(10, PixelKit.sprite("icon_chart"), PixelKit.label("Index Status", "panel-heading"));
        head.setAlignment(Pos.CENTER_LEFT);
        VBox status = new VBox(8, head,
                statRow("icon_doc", "Documents", statDocs),
                statRow("icon_nodes", "Active Vectors", statVectors),
                statRow("icon_stack", "Index Type", statType),
                statRow("icon_chip", "Embedder", statEmbedder));
        status.getStyleClass().add("pixel-panel");

        Node mascotPanel = buildMascotPanel();
        VBox.setVgrow(mascotPanel, Priority.ALWAYS);
        VBox col = new VBox(14, status, mascotPanel);
        PixelKit.fixWidth(col, 330);
        return col;
    }

    private Node statRow(String icon, String key, Label value) {
        HBox row = new HBox(10, PixelKit.sprite(icon), PixelKit.label(key, "stat-key"), PixelKit.hGrow(), value);
        row.setAlignment(Pos.CENTER_LEFT);
        row.getStyleClass().add("stat-row");
        return row;
    }

    private Node buildMascotPanel() {
        Pane decor = new Pane();
        decor.setMouseTransparent(true);
        ImageView c1 = PixelKit.sprite("cloud_small");
        c1.setLayoutX(10);
        c1.setLayoutY(10);
        ImageView c2 = PixelKit.sprite("cloud_big");
        c2.layoutXProperty().bind(decor.widthProperty().subtract(c2.getImage().getWidth() + 6));
        c2.setLayoutY(104);
        decor.getChildren().addAll(c1, c2);

        Region grass = new Region();
        grass.getStyleClass().add("grass-strip");
        PixelKit.fixHeight(grass, 27);

        Pane ground = new Pane();
        ground.setMouseTransparent(true);
        String[] names = {"flower", "sprout_small", "flower_orange", "flower"};
        double[] xs = {0.06, 0.20, 0.80, 0.92};
        for (int i = 0; i < names.length; i++) {
            ImageView v = PixelKit.sprite(names[i]);
            v.layoutXProperty().bind(ground.widthProperty().multiply(xs[i]));
            v.layoutYProperty().bind(ground.heightProperty().subtract(18 + v.getImage().getHeight()));
            ground.getChildren().add(v);
        }

        ImageView mascot = PixelKit.sprite("mascot_190");
        StackPane.setAlignment(mascot, Pos.BOTTOM_LEFT);
        StackPane.setMargin(mascot, new Insets(0, 0, 12, 14));

        bubble.setWrapText(true);
        bubble.setMaxWidth(128);
        ImageView tail = PixelKit.sprite("bubble_tail");
        VBox bubbleBox = new VBox(-3, bubble, tail);
        bubbleBox.setPickOnBounds(false);
        VBox.setMargin(tail, new Insets(0, 0, 0, 22));
        StackPane.setAlignment(bubbleBox, Pos.TOP_RIGHT);
        StackPane.setMargin(bubbleBox, new Insets(22, 6, 0, 0));
        bubbleBox.setMaxSize(Region.USE_PREF_SIZE, Region.USE_PREF_SIZE);

        ImageView sp1 = PixelKit.sprite("sparkle");
        ImageView sp2 = PixelKit.sprite("sparkle");
        StackPane.setAlignment(sp1, Pos.CENTER_LEFT);
        StackPane.setMargin(sp1, new Insets(20, 0, 0, 12));
        StackPane.setAlignment(sp2, Pos.CENTER_RIGHT);
        StackPane.setMargin(sp2, new Insets(40, 18, 0, 0));

        StackPane panel = new StackPane(decor, grass, ground, mascot, sp1, sp2, bubbleBox);
        StackPane.setAlignment(grass, Pos.BOTTOM_CENTER);
        panel.getStyleClass().addAll("pixel-panel", "sky-panel");
        panel.setMinHeight(215);
        return panel;
    }

    // ------------------------------------------------------------------ searching

    void runSearch(String query) {
        SearchBackend b = ctx.backend;
        String q = query == null ? "" : query.trim();
        if (b == null || q.isEmpty()) return;

        closeReader();
        input.setText(q);
        history.remove(q);
        history.addFirst(q);
        while (history.size() > 4) history.removeLast();
        renderChips();

        ctx.mascotLine.set("Searching...");
        matchesLabel.setText("searching...");
        searchButton.setDisable(true);
        int k = ctx.resultCount.get();
        boolean exact = ctx.exactMode.get();

        Task<SearchBackend.Outcome> task = new Task<>() {
            @Override
            protected SearchBackend.Outcome call() throws Exception {
                return b.search(q, k, exact);
            }
        };
        task.setOnSucceeded(e -> {
            searchButton.setDisable(false);
            show(q, task.getValue());
        });
        task.setOnFailed(e -> {
            searchButton.setDisable(false);
            Throwable t = task.getException();
            matchesLabel.setText("search failed");
            ctx.mascotLine.set("Oops, that failed.");
            ctx.toast.accept("Search failed: " + (t == null ? "unknown error" : t.getMessage()));
        });
        ctx.worker.execute(task);
    }

    private void show(String query, SearchBackend.Outcome out) {
        hits = out.hits();
        tokens = TextTools.queryTokens(query);
        int n = hits.size();
        matchesLabel.setText(n + (n == 1 ? " match" : " matches"));
        servedLabel.setText(String.format(Locale.ROOT, "%s | %.1f ms", out.servedBy(), out.millis()));
        if (!phaseOverride) {
            ctx.mascotLine.set(n == 0 ? "Nothing found. Try other words?" : "Found " + n + (n == 1 ? " match!" : " matches!"));
        }
        renderHits();
    }

    private void renderHits() {
        List<SearchBackend.Hit> sorted = new ArrayList<>(hits);
        switch (sortDropdown.value()) {
            case "Document" -> sorted.sort(Comparator.comparing(SearchBackend.Hit::fileName)
                    .thenComparingInt(SearchBackend.Hit::page));
            case "Page" -> sorted.sort(Comparator.comparingInt(SearchBackend.Hit::page)
                    .thenComparing(SearchBackend.Hit::fileName));
            default -> sorted.sort(Comparator.comparingDouble(SearchBackend.Hit::score).reversed());
        }
        cards.clear();
        cardColumn.getChildren().clear();
        selected = -1;
        if (sorted.isEmpty()) {
            showEmpty("No passages matched. Try different words.");
            return;
        }
        for (int i = 0; i < sorted.size(); i++) {
            Card c = new Card(sorted.get(i), i);
            cards.add(c);
            cardColumn.getChildren().add(c);
        }
        select(0);
        scroll.setVvalue(0);
    }

    private void showEmpty(String message) {
        cards.clear();
        selected = -1;
        Label l = PixelKit.label(message, "empty-text");
        l.setWrapText(true);
        VBox box = new VBox(12, PixelKit.sprite("sprout"), l);
        box.setAlignment(Pos.CENTER);
        box.setPadding(new Insets(40, 0, 0, 0));
        cardColumn.getChildren().setAll(box);
    }

    // ------------------------------------------------------------------ selection & keyboard

    private void select(int idx) {
        if (cards.isEmpty()) return;
        int i = Math.max(0, Math.min(cards.size() - 1, idx));
        if (selected >= 0 && selected < cards.size()) cards.get(selected).pseudoClassStateChanged(SELECTED, false);
        selected = i;
        Card c = cards.get(i);
        c.pseudoClassStateChanged(SELECTED, true);
        Platform.runLater(() -> ensureVisible(c));
    }

    private void ensureVisible(Node n) {
        double contentH = cardColumn.getBoundsInLocal().getHeight();
        double viewH = scroll.getViewportBounds().getHeight();
        if (contentH <= viewH) return;
        double top = n.getBoundsInParent().getMinY();
        double bottom = n.getBoundsInParent().getMaxY();
        double scrollTop = scroll.getVvalue() * (contentH - viewH);
        if (top < scrollTop) scroll.setVvalue(top / (contentH - viewH));
        else if (bottom > scrollTop + viewH) scroll.setVvalue((bottom - viewH) / (contentH - viewH));
    }

    private void onKey(KeyEvent e) {
        if (readerPanel.isVisible()) {
            if (e.getCode() == KeyCode.ESCAPE) {
                closeReader();
                e.consume();
            }
            return;
        }
        switch (e.getCode()) {
            case DOWN -> {
                if (!cards.isEmpty()) {
                    scroll.requestFocus();
                    select(selected + 1);
                    e.consume();
                }
            }
            case UP -> {
                if (!cards.isEmpty()) {
                    scroll.requestFocus();
                    select(selected - 1);
                    e.consume();
                }
            }
            case ENTER -> {
                if (scroll.isFocused() && selected >= 0) {
                    openSelected();
                    e.consume();
                }
            }
            case ESCAPE -> {
                input.requestFocus();
                input.selectAll();
                e.consume();
            }
            default -> { }
        }
    }

    // ------------------------------------------------------------------ reader

    void openSelected() {
        if (selected >= 0 && selected < cards.size()) openReader(cards.get(selected).hit);
    }

    private void openReader(SearchBackend.Hit hit) {
        readerHit = hit;
        readerTitle.setText(hit.title());
        readerMeta.setText(String.format(Locale.ROOT, "%s  \u00B7  Page %d  \u00B7  cosine %.3f",
                hit.fileName(), hit.page(), hit.score()));
        TextFlow flow = PixelKit.highlightedFlow(TextTools.collapse(hit.text()), tokens, "reader-text");
        readerBody.getChildren().setAll(flow);
        resultsPanel.setVisible(false);
        readerPanel.setVisible(true);
        readerScroll.setVvalue(0);
    }

    private void closeReader() {
        if (!readerPanel.isVisible()) return;
        readerPanel.setVisible(false);
        resultsPanel.setVisible(true);
        scroll.requestFocus();
    }

    private void openPdf() {
        if (readerHit == null || ctx.host == null) return;
        File f = new File(readerHit.path());
        if (f.isFile()) ctx.host.showDocument(f.toURI().toString());
        else ctx.toast.accept("Could not find " + readerHit.fileName() + " on disk.");
    }

    // ------------------------------------------------------------------ status

    void updateStatus(SearchBackend.IndexInfo info, SearchBackend.HydraInfo h) {
        statDocs.setText(String.valueOf(info.documents()));
        statVectors.setText(info.activeVectors() + " / " + info.storedVectors());
        statType.setText(info.indexType());
        statEmbedder.setText(info.embedder());

        String line = switch (h.phase()) {
            case "COMPETING" -> "Growing new heads...";
            case "TUNING" -> "Fine-tuning...";
            default -> null;
        };
        if (line != null) {
            phaseOverride = true;
            ctx.mascotLine.set(line);
        } else if (phaseOverride) {
            phaseOverride = false;
            ctx.mascotLine.set("Ready to search!");
        }
    }

    // ------------------------------------------------------------------ result card

    private final class Card extends HBox {
        final SearchBackend.Hit hit;

        Card(SearchBackend.Hit hit, int index) {
            this.hit = hit;
            getStyleClass().add("result-card");
            setSpacing(14);
            setAlignment(Pos.TOP_LEFT);

            Label title = PixelKit.label(hit.title(), "card-title");
            title.setWrapText(true);
            Label meta = PixelKit.label(hit.fileName() + "  \u00B7  Page " + hit.page(), "card-meta");
            TextFlow snippet = PixelKit.highlightedFlow(
                    TextTools.snippetAround(hit.text(), tokens, 230), tokens, "snippet-text");
            VBox text = new VBox(4, title, meta, snippet);
            text.setMinWidth(0);
            text.setPrefWidth(10);
            HBox.setHgrow(text, Priority.ALWAYS);

            Label score = PixelKit.label(String.format(Locale.ROOT, "%.3f", hit.score()), "score-pill");
            VBox scoreBox = new VBox(6, score, PixelKit.label("(cosine)", "muted-small"));
            scoreBox.setAlignment(Pos.TOP_CENTER);
            scoreBox.setMinWidth(Region.USE_PREF_SIZE);

            ImageView icon = PixelKit.sprite("doc_card");
            VBox iconBox = new VBox(icon);
            iconBox.setAlignment(Pos.TOP_CENTER);

            getChildren().addAll(iconBox, text, scoreBox);
            setOnMouseClicked(e -> {
                scroll.requestFocus();
                select(index);
                if (e.getClickCount() == 2) openReader(hit);
            });
        }
    }
}
