package macro;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Vector;

import star.base.neo.NamedObject;
import star.common.Boundary;
import star.common.FieldFunction;
import star.common.Region;
import star.common.Simulation;
import star.common.StarMacro;
import star.common.Table;
import star.common.XyzInternalTable;

/**
 * Auto-export helper for STAR-CCM+ -> GeoPT.
 *
 * This macro creates temporary internal tables automatically:
 * - volume: first region in the simulation
 * - surface: boundary named "Hull", otherwise the first wall-like boundary
 *
 * It exports CSV files under ./geopt_exports next to the .sim file, with
 * component-wise columns that the Python converter can parse.
 */
public class AutoExportGeoPTFields extends StarMacro {

    private static final String OUTPUT_SUBDIR = "geopt_exports";
    private static final String VOLUME_TABLE_NAME = "GeoPT_Volume_Table";
    private static final String SURFACE_TABLE_NAME = "GeoPT_Surface_Table";

    @Override
    public void execute() {
        Simulation sim = getActiveSimulation();

        Region volumeRegion = pickVolumeRegion(sim);
        Boundary hullBoundary = pickHullBoundary(volumeRegion);

        sim.println("[GeoPT] Volume region: " + volumeRegion.getPresentationName());
        sim.println("[GeoPT] Surface boundary: " + hullBoundary.getPresentationName());

        XyzInternalTable volumeTable = createTable(
            sim,
            VOLUME_TABLE_NAME,
            asNamedObjects(volumeRegion),
            volumeFunctions(sim));

        XyzInternalTable surfaceTable = createTable(
            sim,
            SURFACE_TABLE_NAME,
            asNamedObjects(hullBoundary),
            surfaceFunctions(sim));

        File outputDir = new File(sim.getSessionDirFile(), OUTPUT_SUBDIR);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new RuntimeException("Cannot create output directory: " + outputDir.getAbsolutePath());
        }

        String simName = sim.getPresentationName().replaceAll("\\s+", "_");
        File volumeCsv = new File(outputDir, simName + "_volume.csv");
        File surfaceCsv = new File(outputDir, simName + "_surface.csv");

        volumeTable.extract();
        volumeTable.export(volumeCsv.getAbsolutePath(), ",");
        surfaceTable.extract();
        surfaceTable.export(surfaceCsv.getAbsolutePath(), ",");

        sim.println("[GeoPT] Exported volume CSV: " + volumeCsv.getAbsolutePath());
        sim.println("[GeoPT] Exported surface CSV: " + surfaceCsv.getAbsolutePath());
    }

    private Region pickVolumeRegion(Simulation sim) {
        Collection<Region> regions = sim.getRegionManager().getRegions();
        if (regions.isEmpty()) {
            throw new IllegalStateException("No regions found in simulation.");
        }
        return regions.iterator().next();
    }

    private Boundary pickHullBoundary(Region region) {
        Boundary namedHull = region.getBoundaryManager().getBoundary("Hull");
        if (namedHull != null) {
            return namedHull;
        }

        for (Boundary boundary : region.getBoundaryManager().getBoundaries()) {
            if (boundary.getBoundaryType().getClass().getSimpleName().contains("Wall")) {
                return boundary;
            }
        }

        throw new IllegalStateException("No wall-like boundary found for surface export.");
    }

    private XyzInternalTable createTable(
        Simulation sim,
        String tableName,
        Collection<NamedObject> objects,
        Vector<FieldFunction> functions
    ) {
        Table existing = null;
        for (Table table : sim.getTableManager().getObjects()) {
            if (tableName.equals(table.getPresentationName())) {
                existing = table;
                break;
            }
        }
        if (existing != null) {
            sim.getTableManager().deleteChildren(java.util.Collections.singleton(existing));
        }

        XyzInternalTable table = sim.getTableManager().createInternal(XyzInternalTable.class);
        table.setPresentationName(tableName);
        table.setObjects(objects);
        table.setFieldFunctions(functions);
        table.setExtractVertexData(false);
        return table;
    }

    private Collection<NamedObject> asNamedObjects(NamedObject object) {
        ArrayList<NamedObject> objects = new ArrayList<>();
        objects.add(object);
        return objects;
    }

    private Vector<FieldFunction> volumeFunctions(Simulation sim) {
        Vector<FieldFunction> functions = new Vector<>();
        addVectorComponents(functions, requireFieldFunction(sim, "Position"));
        addVectorComponents(functions, requireFieldFunction(sim, "Velocity"));
        functions.add(requireFieldFunction(sim, "Pressure"));
        return functions;
    }

    private Vector<FieldFunction> surfaceFunctions(Simulation sim) {
        Vector<FieldFunction> functions = new Vector<>();
        addVectorComponents(functions, requireFieldFunction(sim, "Position"));
        addVectorComponents(functions, requireFieldFunction(sim, "Velocity"));
        functions.add(requireFieldFunction(sim, "Pressure"));
        addVectorComponents(functions, requireFieldFunction(sim, "Normal"));
        return functions;
    }

    private void addVectorComponents(Vector<FieldFunction> functions, FieldFunction vectorField) {
        for (int i = 0; i < 3; i++) {
            functions.add(vectorField.getComponentFunction(i));
        }
    }

    private FieldFunction requireFieldFunction(Simulation sim, String name) {
        FieldFunction ff = sim.getFieldFunctionManager().getFunction(name);
        if (ff == null) {
            throw new IllegalStateException("Missing field function: " + name);
        }
        return ff;
    }
}
