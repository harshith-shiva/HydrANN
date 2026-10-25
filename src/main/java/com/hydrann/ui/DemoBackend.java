package com.hydrann.ui;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * A self-contained backend with a handful of sample passages, so the UI can be explored
 * (and screenshotted) without the embedding model or any PDFs. Start it with {@code Launcher demo}.
 */
final class DemoBackend implements SearchBackend {

    private record Passage(String file, int page, String text) {}

    private static final String DIR = "C:\\Users\\you\\hydrann\\data\\raw-docs\\";

    private static final List<Passage> PASSAGES = List.of(
            new Passage("ncert_science.pdf", 14,
                    "50 BIOLOGY\nlungs. Air sacs connected to lungs supplement respiration. Sexes are separate. Fertilisation is "
                            + "internal. Birds are warm-blooded animals with feathers on their bodies and forelimbs modified into wings "
                            + "for flying. Examples : Corvus (Crow), Columba (Pigeon), Psittacula (Parrot), Struthio (Ostrich)."),
            new Passage("ncert_science.pdf", 17,
                    "ANIMAL KINGDOM 53\nAves are warm-blooded animals with feathers on their bodies and forelimbs modified into wings "
                            + "for flying. Hind limbs are adapted for walking, swimming, perching or clasping. What are the modifications "
                            + "that are observed in birds that help them fly?"),
            new Passage("ncert_science.pdf", 9,
                    "Malaria is caused by a protozoan parasite carried by the female Anopheles mosquito. The disease spreads when an "
                            + "infected mosquito bites a person and the parasite enters the bloodstream and multiplies in the liver."),
            new Passage("200-Science-Facts-to-Know.pdf", 2,
                    "Natural selection is the differential reproductive success of members of a species based on differences in "
                            + "phenotype. The mechanism of natural selection can also be explained by the phrase survival of the fittest."),
            new Passage("200-Science-Facts-to-Know.pdf", 2,
                    "In 1859, Charles Darwin published On the Origin of Species where he outlined his theory of evolution by natural "
                            + "selection. He observed 13 different varieties of finches in the Galapagos islands."),
            new Passage("200-Science-Facts-to-Know.pdf", 9,
                    "Gravity is the weakest of the fundamental forces, which causes objects to fall to the ground."),
            new Passage("physics-motion-notes.pdf", 4,
                    "Newton's laws of motion: an object stays at rest or in uniform motion unless acted on by a net force; force equals "
                            + "mass times acceleration; every action has an equal and opposite reaction."),
            new Passage("ncert_science.pdf", 22,
                    "Photosynthesis is the process by which green plants make food. Chlorophyll in the leaves absorbs sunlight, and "
                            + "carbon dioxide and water are converted into glucose and oxygen."),
            new Passage("ncert_science.pdf", 31,
                    "Mosquitoes breed in stagnant water. Preventing standing water and using bed nets reduces the spread of "
                            + "malaria and dengue in a community."),
            new Passage("physics-motion-notes.pdf", 7,
                    "Free fall: all objects near the surface of the earth accelerate downward at about 9.8 metres per second squared "
                            + "when air resistance is ignored, so a heavy and a light object land together."));

    private Consumer<String> listener = s -> {};

    @Override
    public Outcome search(String query, int k, boolean exact) {
        List<String> tokens = TextTools.queryTokens(query);
        List<Hit> hits = new ArrayList<>();
        for (Passage p : PASSAGES) {
            String lower = p.text().toLowerCase(Locale.ROOT);
            int matched = 0;
            for (String t : tokens) {
                String stem = t.length() > 5 ? t.substring(0, t.length() - 1) : t;
                if (lower.contains(stem)) matched++;
            }
            double ratio = tokens.isEmpty() ? 0 : (double) matched / tokens.size();
            double noise = ((p.text().hashCode() & 0xff) / 255.0) * 0.06;
            double score = 0.14 + 0.62 * ratio + noise;
            if (matched == 0 && score > 0.2) score = 0.17 + noise * 0.5;
            String name = TextTools.fileName(p.file());
            hits.add(new Hit(TextTools.deriveTitle(p.text(), name), name, DIR + p.file(), p.page(), 0, score, p.text()));
        }
        hits.sort(Comparator.comparingDouble(Hit::score).reversed());
        List<Hit> top = new ArrayList<>(hits.subList(0, Math.min(k, hits.size())));
        return new Outcome(top, exact ? 1.8 : 0.9, exact ? "Exact scan" : "HNSW");
    }

    @Override
    public IndexInfo indexInfo() {
        return new IndexInfo(3, 152, 450, "HNSW", "hnsw(M=16,efC=100,efS=50)", "MiniLM (384d)");
    }

    @Override
    public List<DocInfo> documents() {
        return List.of(
                new DocInfo("200-Science-Facts-to-Know.pdf", DIR + "200-Science-Facts-to-Know.pdf", 31, 11),
                new DocInfo("ncert_science.pdf", DIR + "ncert_science.pdf", 94, 22),
                new DocInfo("physics-motion-notes.pdf", DIR + "physics-motion-notes.pdf", 27, 9));
    }

    @Override
    public HydraInfo hydra() {
        long now = System.currentTimeMillis();
        return new HydraInfo("STABLE", "", "hnsw(M=16,efC=100,efS=50)", "hnsw",
                152, 450, 20, 0.97, 410, 780, 2300, 5.6,
                186_000, now - 14 * 60_000L, 1, 2, 1,
                "SWITCH brute-force -> hnsw(M=16,efC=100,efS=50) (score 0.412 vs 0.988)", now - 12 * 60_000L,
                List.of("> hnsw(M=16,efC=100,efS=50)       recall 0.97 | avg 410 us | mem 181 KB | build 120 ms | score 0.412",
                        "  brute-force (incumbent)         recall 1.00 | avg 2,300 us | mem 0 B | build 0 ms | score 0.988",
                        "  ivf(nClusters=12,nProbe=3)      recall 0.62 | DISQUALIFIED (recall < 0.90)",
                        "  lsh                             recall 0.58 | DISQUALIFIED (recall < 0.90)"),
                List.of("12:01:14  Hydra online. Active head: brute-force",
                        "12:01:16  COMPETITION started: initial evaluation on 152 active vectors",
                        "12:01:19  COMPETITION finished in 3s: SWITCH brute-force -> hnsw",
                        "12:09:44  TUNE hnsw: efSearch 50 -> 75 (recall 0.88 < target 0.90)"));
    }

    @Override
    public void rescan() {
        // nothing to scan in demo mode
    }

    @Override
    public void setEventListener(Consumer<String> listener) {
        this.listener = listener;
    }

    @Override
    public String libraryFolder() {
        return "data/raw-docs";
    }

    @Override
    public void close() {
        // nothing to release
    }
}
