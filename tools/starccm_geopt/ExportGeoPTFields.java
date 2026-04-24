package macro;

import java.io.File;

import star.base.neo.*;
import star.common.*;
import star.vis.*;

/**
 * STAR-CCM+ macro template:
 * 1) Prepare one volume table and one surface table in your .sim case.
 * 2) Put pressure/velocity/coordinates (and optional normals) into those tables.
 * 3) Run this macro to export CSV files for downstream GeoPT conversion.
 *
 * Version notes:
 * - Tested as a template against common STAR-CCM+ table APIs.
 * - Table names/columns are project-dependent and can be edited below.
 */
public class ExportGeoPTFields extends StarMacro {

    // === You usually only need to edit these constants ===
    private static final String VOLUME_TABLE_NAME = "GeoPT_Volume_Table";
    private static final String SURFACE_TABLE_NAME = "GeoPT_Surface_Table";
    private static final String OUTPUT_SUBDIR = "geopt_exports";

    @Override
    public void execute() {
        Simulation sim = getActiveSimulation();

        String simName = sim.getPresentationName().replaceAll("\\s+", "_");
        File outputDir = new File(sim.getSessionDirFile(), OUTPUT_SUBDIR);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new RuntimeException("Cannot create output directory: " + outputDir.getAbsolutePath());
        }

        exportTableCsv(sim, VOLUME_TABLE_NAME, new File(outputDir, simName + "_volume.csv"));
        exportTableCsv(sim, SURFACE_TABLE_NAME, new File(outputDir, simName + "_surface.csv"));

        sim.println("[GeoPT] Export finished: " + outputDir.getAbsolutePath());
    }

    private void exportTableCsv(Simulation sim, String tableName, File outFile) {
        ClientServerObject cso = sim.getTableManager().getObject(tableName);
        if (cso == null) {
            throw new IllegalArgumentException("Table not found: " + tableName);
        }
        if (!(cso instanceof Table)) {
            throw new IllegalArgumentException("Object is not a table: " + tableName);
        }

        Table table = (Table) cso;
        table.extract();
        table.export(outFile.getAbsolutePath(), ",");

        sim.println("[GeoPT] Exported table '" + tableName + "' -> " + outFile.getAbsolutePath());
    }
}
