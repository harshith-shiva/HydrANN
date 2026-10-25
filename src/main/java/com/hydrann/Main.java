package com.hydrann;

import com.hydrann.annheads.ANNHead;
import com.hydrann.annheads.BruteForceHead;
import com.hydrann.annheads.HnswHead;
// import com.hydrann.annheads.IvfHead;
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

    
    }

    // -------------------------------------------------------------------------

}  