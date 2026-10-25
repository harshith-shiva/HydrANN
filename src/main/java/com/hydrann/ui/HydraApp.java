package com.hydrann.ui;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.PauseTransition;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.css.PseudoClass;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Cursor;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/** The HydrANN window: custom pixel title bar, sidebar navigation and the four views. */
public class HydraApp extends Application {

    /** Used by UI tests to drive the app once the backend is up. */
    static volatile Consumer<HydraApp> testHook;

    private static final PseudoClass ON = PseudoClass.getPseudoClass("on");

    private final ExecutorService worker = Executors.newSingleThreadExecutor(daemon("ui-worker"));
    private final ExecutorService statusWorker = Executors.newSingleThreadExecutor(daemon("ui-status"));
    private final AppContext ctx = new AppContext(worker, statusWorker);

    private Stage stage;
    private HomeView home;
    private LibraryView library;
    private IndexLabView lab;
    private SettingsView settings;

    private final StackPane viewHost = new StackPane();
    private final ToggleGroup navGroup = new ToggleGroup();
    private final Label toast = new Label();
    private final PauseTransition toastTimer = new PauseTransition(Duration.seconds(6));
    private String currentTarget = "home";
    private Timeline poller;

    @Override
    public void start(Stage primary) {
        this.stage = primary;
        PixelKit.loadFonts();

        boolean demo = getParameters() != null && getParameters().getRaw().contains("demo");
        ctx.host = getHostServices();
        ctx.toast = this::showToast;
        ctx.rescan = this::rescan;

        home = new HomeView(ctx);
        library = new LibraryView(ctx);
        lab = new IndexLabView();
        settings = new SettingsView(ctx);

        // ---- loading overlay
        Label loadingText = PixelKit.label("Waking the hydra...", "loading-text");
        HBox blocks = new HBox(8);
        blocks.setAlignment(Pos.CENTER);
        List<Region> blockNodes = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            Region r = new Region();
            r.getStyleClass().add("load-block");
            blockNodes.add(r);
            blocks.getChildren().add(r);
        }
        int[] tick = {0};
        Timeline blink = new Timeline(new KeyFrame(Duration.millis(220), e -> {
            for (int i = 0; i < blockNodes.size(); i++) blockNodes.get(i).pseudoClassStateChanged(ON, i == tick[0] % blockNodes.size());
            tick[0]++;
        }));
        blink.setCycleCount(Animation.INDEFINITE);
        blink.play();
        VBox loading = new VBox(16, PixelKit.sprite("mascot_230"), PixelKit.label("HydrANN", "brand-title"), loadingText, blocks);
        loading.setAlignment(Pos.CENTER);
        loading.getStyleClass().add("loading-overlay");

        // ---- frame
        StackPane center = new StackPane(viewHost, loading);
        center.setMinSize(0, 0);
        viewHost.setMinSize(0, 0);
        viewHost.setAlignment(Pos.TOP_LEFT);
        javafx.scene.shape.Rectangle clip = new javafx.scene.shape.Rectangle();
        clip.widthProperty().bind(center.widthProperty());
        clip.heightProperty().bind(center.heightProperty());
        center.setClip(clip);
        BorderPane frame = new BorderPane();
        frame.getStyleClass().add("window-frame");
        frame.setTop(buildTitleBar());
        frame.setLeft(buildSidebar());
        frame.setCenter(center);
        viewHost.getStyleClass().add("main-area");
        navigate("home");

        toast.getStyleClass().add("toast");
        toast.setWrapText(true);
        toast.setMaxWidth(560);
        toast.setVisible(false);
        toast.setMouseTransparent(true);
        StackPane.setAlignment(toast, Pos.BOTTOM_CENTER);
        StackPane.setMargin(toast, new Insets(0, 0, 28, 0));
        toastTimer.setOnFinished(e -> toast.setVisible(false));

        StackPane grip = new StackPane(PixelKit.sprite("grip"));
        grip.setCursor(Cursor.SE_RESIZE);
        grip.setMaxSize(24, 24);
        StackPane.setAlignment(grip, Pos.BOTTOM_RIGHT);
        grip.setOnMouseDragged(e -> {
            stage.setWidth(Math.max(stage.getMinWidth(), e.getScreenX() - stage.getX() + 8));
            stage.setHeight(Math.max(stage.getMinHeight(), e.getScreenY() - stage.getY() + 8));
        });

        StackPane root = new StackPane(frame, toast, grip);
        javafx.geometry.Rectangle2D screen = javafx.stage.Screen.getPrimary().getVisualBounds();
        Scene scene = new Scene(root, Math.min(1320, screen.getWidth() - 40), Math.min(900, screen.getHeight() - 40));
        scene.getStylesheets().add(PixelKit.stylesheet());

        stage.initStyle(StageStyle.UNDECORATED);
        stage.setTitle("HydrANN");
        stage.getIcons().add(PixelKit.image("mascot_44"));
        stage.setMinWidth(Math.min(1060, screen.getWidth() - 40));
        stage.setMinHeight(Math.min(740, screen.getHeight() - 40));
        stage.setScene(scene);
        stage.show();

        // ---- open the backend off the UI thread
        Task<SearchBackend> open = new Task<>() {
            @Override
            protected SearchBackend call() throws Exception {
                if (demo) return new DemoBackend();
                return LiveBackend.open(msg -> updateMessage(msg));
            }
        };
        loadingText.textProperty().bind(open.messageProperty());
        open.setOnSucceeded(e -> {
            loadingText.textProperty().unbind();
            blink.stop();
            ctx.backend = open.getValue();
            ctx.backend.setEventListener(msg -> Platform.runLater(() -> onHydraEvent(msg)));
            center.getChildren().remove(loading);
            ctx.mascotLine.set("Ready to search!");
            home.focusSearch();
            startPolling();
            Consumer<HydraApp> hook = testHook;
            if (hook != null) hook.accept(this);
        });
        open.setOnFailed(e -> {
            loadingText.textProperty().unbind();
            blink.stop();
            Throwable t = open.getException();
            if (t != null) t.printStackTrace();
            loadingText.setText("Could not start: " + (t == null ? "unknown error" : t.getMessage()));
            loadingText.setWrapText(true);
            loadingText.setMaxWidth(520);
        });
        worker.execute(open);
    }

    // ------------------------------------------------------------------ chrome

    private Node buildTitleBar() {
        Label name = PixelKit.label("HydrANN", "title-bar-text");
        HBox bar = new HBox(10, PixelKit.sprite("mascot_28"), name, PixelKit.hGrow(),
                winButton("win_min", () -> stage.setIconified(true)),
                winButton("win_max", this::toggleMaximize),
                winButton("win_close", Platform::exit));
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.getStyleClass().add("title-bar");

        double[] off = new double[2];
        bar.setOnMousePressed(e -> {
            off[0] = e.getScreenX() - stage.getX();
            off[1] = e.getScreenY() - stage.getY();
        });
        bar.setOnMouseDragged(e -> {
            if (!stage.isMaximized()) {
                stage.setX(e.getScreenX() - off[0]);
                stage.setY(e.getScreenY() - off[1]);
            }
        });
        bar.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) toggleMaximize();
        });
        return bar;
    }

    private Button winButton(String sprite, Runnable action) {
        Button b = new Button();
        b.setGraphic(PixelKit.sprite(sprite));
        b.getStyleClass().add("win-button");
        b.setFocusTraversable(false);
        b.setOnAction(e -> action.run());
        return b;
    }

    private void toggleMaximize() {
        stage.setMaximized(!stage.isMaximized());
    }

    private Node buildSidebar() {
        VBox nav = new VBox(8,
                navItem("Home", "home", "home"),
                navItem("Library", "folder", "library"),
                navItem("Index Lab", "nodes", "lab"),
                navItem("Settings", "gear", "settings"));
        nav.setPadding(new Insets(18, 12, 0, 12));
        navGroup.selectedToggleProperty().addListener((o, was, now) -> {
            if (now == null && was != null) was.setSelected(true);   // one item always stays selected
        });

        VBox side = new VBox(nav, PixelKit.vGrow(), buildSidebarDecor());
        side.getStyleClass().add("sidebar");
        PixelKit.fixWidth(side, 200);
        return side;
    }

    private ToggleButton navItem(String text, String icon, String target) {
        ToggleButton b = new ToggleButton(text);
        b.getStyleClass().add("nav-item");
        b.setToggleGroup(navGroup);
        b.setUserData(target);
        b.setMaxWidth(Double.MAX_VALUE);
        ImageView dark = PixelKit.sprite("icon_" + icon);
        ImageView light = PixelKit.sprite("icon_" + icon + "_light");
        b.setGraphic(dark);
        b.selectedProperty().addListener((o, was, now) -> b.setGraphic(now ? light : dark));
        b.setOnAction(e -> navigate(target));
        return b;
    }

    private Node buildSidebarDecor() {
        Pane p = new Pane();
        PixelKit.fixHeight(p, 300);

        ImageView cloud1 = PixelKit.sprite("cloud_small");
        cloud1.setLayoutX(18);
        cloud1.setLayoutY(2);
        ImageView cloud2 = PixelKit.sprite("cloud_small");
        cloud2.setLayoutX(118);
        cloud2.setLayoutY(52);

        ImageView sproutL = PixelKit.sprite("sprout");
        sproutL.setLayoutX(22);
        sproutL.setLayoutY(74);
        ImageView sproutR = PixelKit.sprite("sprout");
        sproutR.setLayoutX(138);
        sproutR.setLayoutY(70);

        Region legL = new Region();
        legL.getStyleClass().add("sign-leg");
        legL.setPrefSize(10, 70);
        legL.setLayoutX(50);
        legL.setLayoutY(190);
        Region legR = new Region();
        legR.getStyleClass().add("sign-leg");
        legR.setPrefSize(10, 70);
        legR.setLayoutX(138);
        legR.setLayoutY(190);

        Label signText = new Label("LOCAL\nSEARCH\nENGINE");
        signText.getStyleClass().add("sign-text");
        StackPane sign = new StackPane(signText);
        sign.getStyleClass().add("sign-board");
        sign.setPrefSize(150, 96);
        sign.setLayoutX(24);
        sign.setLayoutY(116);

        Region grass = new Region();
        grass.getStyleClass().add("grass-strip");
        grass.prefWidthProperty().bind(p.widthProperty());
        grass.setPrefHeight(27);
        grass.setLayoutY(273);

        ImageView bushL = PixelKit.sprite("bush");
        bushL.setLayoutX(-22);
        bushL.setLayoutY(224);
        ImageView bushR = PixelKit.sprite("bush_small");
        bushR.setLayoutX(128);
        bushR.setLayoutY(244);
        ImageView flower = PixelKit.sprite("flower");
        flower.setLayoutX(104);
        flower.setLayoutY(255);

        p.getChildren().addAll(cloud1, cloud2, sproutL, sproutR, legL, legR, sign, grass, bushL, bushR, flower);
        p.setMouseTransparent(true);
        return p;
    }

    // ------------------------------------------------------------------ navigation

    void navigate(String target) {
        currentTarget = target;
        Node v = switch (target) {
            case "library" -> library;
            case "lab" -> lab;
            case "settings" -> settings;
            default -> home;
        };
        viewHost.getChildren().setAll(v);
        for (Toggle t : navGroup.getToggles()) {
            if (target.equals(t.getUserData()) && !t.isSelected()) t.setSelected(true);
        }
        if (target.equals("home")) home.focusSearch();
        if (target.equals("library")) poll();
    }

    // ------------------------------------------------------------------ status polling

    private void startPolling() {
        poller = new Timeline(new KeyFrame(Duration.seconds(1), e -> poll()));
        poller.setCycleCount(Animation.INDEFINITE);
        poller.play();
        poll();
    }

    private void poll() {
        SearchBackend b = ctx.backend;
        if (b == null) return;
        boolean wantDocs = currentTarget.equals("library");
        statusWorker.execute(() -> {
            try {
                SearchBackend.IndexInfo info = b.indexInfo();
                SearchBackend.HydraInfo h = b.hydra();
                List<SearchBackend.DocInfo> docs = wantDocs ? b.documents() : null;
                Platform.runLater(() -> {
                    home.updateStatus(info, h);
                    lab.update(h);
                    if (docs != null) library.update(docs);
                });
            } catch (RuntimeException ignored) {
                // a transient read while the index is changing; the next tick retries
            }
        });
    }

    private void onHydraEvent(String msg) {
        String first = msg.lines().findFirst().orElse(msg);
        showToast("Hydra: " + first.replaceFirst("^\\d{2}:\\d{2}:\\d{2}\\s+", ""));
        poll();
    }

    private void rescan() {
        SearchBackend b = ctx.backend;
        if (b == null) return;
        showToast("Scanning your library for changes...");
        worker.execute(() -> {
            try {
                b.rescan();
                Platform.runLater(() -> {
                    showToast("Library is up to date.");
                    poll();
                });
            } catch (Exception e) {
                Platform.runLater(() -> showToast("Scan failed: " + e.getMessage()));
            }
        });
    }

    private void showToast(String text) {
        toast.setText(text);
        toast.setVisible(true);
        toastTimer.playFromStart();
    }

    // ------------------------------------------------------------------ test access

    Stage stage() {
        return stage;
    }

    HomeView home() {
        return home;
    }

    @Override
    public void stop() {
        if (poller != null) poller.stop();
        worker.shutdownNow();
        statusWorker.shutdownNow();
        SearchBackend b = ctx.backend;
        if (b != null) b.close();
    }

    private static java.util.concurrent.ThreadFactory daemon(String name) {
        return r -> {
            Thread t = new Thread(r, name);
            t.setDaemon(true);
            return t;
        };
    }
}
