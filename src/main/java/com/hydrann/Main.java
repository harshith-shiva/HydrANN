package com.hydrann;

import com.hydrann.annheads.ANNHead;
import com.hydrann.annheads.BruteForceHead;
import com.hydrann.annheads.HnswHead;
import com.hydrann.annheads.IvfHead;
import com.hydrann.ingestion.ExtractedPage;
import com.hydrann.ingestion.ExtractedPdf;
import com.hydrann.ingestion.PdfExtractor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // PART 1 — PDF Extraction (existing functionality)
        // =====================================================================

        Path pdfPath = Path.of("data/raw-docs/Lab Exercise-2_OR DB.pdf");
        PdfExtractor extractor = new PdfExtractor();
        ExtractedPdf pdf = extractor.extract(pdfPath);

        System.out.println("===== PDF INFORMATION =====");
        System.out.println("Document ID : " + pdf.getDocumentId());
        System.out.println("File name   : " + pdf.getFileName());
        System.out.println("File path   : " + pdf.getFilePath());
        System.out.println("Title       : " + pdf.getTitle());
        System.out.println("Author      : " + pdf.getAuthor());
        System.out.println("Subject     : " + pdf.getSubject());
        System.out.println("Page count  : " + pdf.getPageCount());

        System.out.println("\n===== EXTRACTED TEXT =====");
        for (ExtractedPage page : pdf.getPages()) {
            System.out.println("\n----- PDF PAGE " + page.getPdfPageNumber() + " -----");
            System.out.println(page.getText());
        }

        // =====================================================================
        // PART 2 — ANN Head Demo (new functionality)
        // =====================================================================

        System.out.println("\n\n===== ANN HEAD DEMO =====");

        // Build a tiny synthetic corpus of 3-D vectors
        List<float[]> corpus = new ArrayList<>();
        corpus.add(new float[]{ 1.0f,  0.0f,  0.0f });  // 0: +x axis
        corpus.add(new float[]{ 0.9f,  0.1f,  0.0f });  // 1: near +x
        corpus.add(new float[]{ 0.0f,  1.0f,  0.0f });  // 2: +y axis
        corpus.add(new float[]{ 0.0f,  0.0f,  1.0f });  // 3: +z axis
        corpus.add(new float[]{-1.0f,  0.0f,  0.0f });  // 4: -x axis
        corpus.add(new float[]{ 0.5f,  0.5f,  0.0f });  // 5: diagonal xy
        corpus.add(new float[]{ 0.0f, -1.0f,  0.0f });  // 6: -y axis

        float[] query = new float[]{ 1.0f, 0.05f, 0.0f }; // very close to +x

        System.out.println("Query vector : " + Arrays.toString(query));
        System.out.println("Searching for k=3 nearest neighbours\n");

        runHead(new BruteForceHead(),        corpus, query, 3);
        runHead(new IvfHead(3, 2, 25, 7L),   corpus, query, 3);
        runHead(new HnswHead(4, 50, 20, 7L), corpus, query, 3);
    }

    // -------------------------------------------------------------------------

    private static void runHead(ANNHead head, List<float[]> corpus,
                                 float[] query, int k) {
        head.build(corpus);
        List<Integer> result = head.search(query, k);

        System.out.printf("[%s] Top-%d: %s%n", head.name(), k, result);
        for (int idx : result) {
            System.out.printf("          idx=%-2d  %s%n",
                    idx, Arrays.toString(corpus.get(idx)));
        }
        System.out.println();
    }
}