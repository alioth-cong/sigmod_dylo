/*
 * BaselineDriver — uniform CLI around the dynamo (nogrady) algorithms:
 *   louvain-static, qca, gremod, batch
 *
 * Invocation (insertion-only stream):
 *   BaselineDriver louvain-static <initial.txt> <inc-base/inc.txt> <numBatches> <outDir>
 *   BaselineDriver qca            <initial.txt> <initial-com.txt> <inc-base/inc.txt> <numBatches> <outDir>
 *   BaselineDriver gremod         <initial.txt> <initial-com.txt> <inc-base/inc.txt> <numBatches> <outDir>
 *   BaselineDriver batch          <initial.txt> <initial-com.txt> <inc-base/inc.txt> <numBatches> <outDir>
 *
 * `inc-base/inc.txt` is a logical path; the algorithms read inc_1.txt, inc_2.txt, ... (via dynamo's
 * FileUtil.extendFileName), so the actual files in the inc directory must be inc_1.txt, inc_2.txt, ...
 *
 * Produces in <outDir>:
 *   metrics.tsv     batch<TAB>latency_ms<TAB>modularity<TAB>num_communities
 *   com_<k>.txt     community structure after batch k (dynamo native format —
 *                   one community per line, tab-separated node IDs)
 */
import java.io.*;
import java.util.*;
import org.dzhuang.dynamic.OtherAlgorithms.Louvain;
import org.dzhuang.dynamic.OtherAlgorithms.QCA;
import org.dzhuang.dynamic.OtherAlgorithms.GreMod;
import org.dzhuang.dynamic.OtherAlgorithms.BatchInc;
import org.dzhuang.dynamic.DynaMo.Network;
import org.dzhuang.dynamic.DynaMo.ModularityOptimizer_DynaMo;

public class BaselineDriver {

    static void appendMetric(PrintWriter pw, int batch, long latencyMs, double modularity, int numComms) {
        pw.println(batch + "\t" + latencyMs + "\t" + modularity + "\t" + numComms);
        pw.flush();
    }

    /** Mirror of dynamo's FileUtil.extendFileName. */
    static String extendFileName(String name, String extend) {
        int idx = name.lastIndexOf('.');
        return name.substring(0, idx) + extend + "." + name.substring(idx + 1);
    }

    static void copyFile(String src, String dst) throws IOException {
        try (BufferedReader r = new BufferedReader(new FileReader(src));
             PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(dst)))) {
            String line; while ((line = r.readLine()) != null) pw.println(line);
        }
    }

    /** Build initial.txt + inc_1..inc_k into one cumulative edge-list (u<TAB>v<TAB>1.0). */
    static String makeCumulativeGraph(String initialPath, String incBase, int upToK, String outDir) throws IOException {
        File f = new File(outDir, "cumulative_" + upToK + ".txt");
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(f)))) {
            try (BufferedReader r = new BufferedReader(new FileReader(initialPath))) {
                String line; while ((line = r.readLine()) != null) pw.println(line);
            }
            for (int i = 1; i <= upToK; i++) {
                String inc = extendFileName(incBase, "_" + i);
                try (BufferedReader r = new BufferedReader(new FileReader(inc))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        String[] t = line.split("\t");
                        if (t.length >= 2) pw.println(t[0] + "\t" + t[1] + "\t1.0");
                    }
                }
            }
        }
        return f.getAbsolutePath();
    }

    /** louvain-static: per-batch rebuild the cumulative graph, run Louvain from scratch.
     *  Writes one community file per batch.
     */
    static void runLouvainStatic(String initialPath, String incBase, int numBatches, String outDir) throws Exception {
        new File(outDir).mkdirs();
        PrintWriter metrics = new PrintWriter(new BufferedWriter(new FileWriter(new File(outDir, "metrics.tsv"))));
        metrics.println("batch\tlatency_ms\tmodularity\tnum_communities");

        // Batch 0 — initial graph clustering
        long t0a = System.currentTimeMillis();
        Louvain l0 = new Louvain();
        Louvain c0 = l0.runAndExport(initialPath, 0.0001, new File(outDir, "com_0.txt").getAbsolutePath());
        long t0b = System.currentTimeMillis();
        appendMetric(metrics, 0, t0b - t0a, c0.modularity(), c0.nonEmptyCommunities());

        for (int k = 1; k <= numBatches; k++) {
            String cumPath = makeCumulativeGraph(initialPath, incBase, k, outDir);
            long t1 = System.currentTimeMillis();
            Louvain L = new Louvain();
            Louvain c = L.runAndExport(cumPath, 0.0001, new File(outDir, "com_" + k + ".txt").getAbsolutePath());
            long t2 = System.currentTimeMillis();
            appendMetric(metrics, k, t2 - t1, c.modularity(), c.nonEmptyCommunities());
            new File(cumPath).delete();
        }
        metrics.close();
    }

    /** Drive QCA/GreMod/BatchInc in one shot — they iterate over inc_1..inc_K internally
     *  and report modList/timeList per batch.
     */
    static void runIncremental(String which, String initialPath, String initialComPath,
                               String incBase, int numBatches, String outDir) throws Exception {
        new File(outDir).mkdirs();
        PrintWriter metrics = new PrintWriter(new BufferedWriter(new FileWriter(new File(outDir, "metrics.tsv"))));
        metrics.println("batch\tlatency_ms\tmodularity\tnum_communities");

        String comOutBase = new File(outDir, "com.txt").getAbsolutePath();
        HashMap result;
        double initMod;

        if (which.equals("qca")) {
            QCA qca = new QCA();
            qca.init(initialPath, initialComPath, 0.0001);
            initMod = qca.modularity();
            appendMetric(metrics, 0, 0, initMod, qca.nonEmptyCommunities());
            copyFile(initialComPath, new File(outDir, "com_0.txt").getAbsolutePath());
            result = qca.increase(incBase, numBatches, comOutBase);
        } else if (which.equals("gremod")) {
            GreMod gm = new GreMod();
            gm.initialize(initialPath, initialComPath);
            initMod = gm.modularity();
            appendMetric(metrics, 0, 0, initMod, -1);
            copyFile(initialComPath, new File(outDir, "com_0.txt").getAbsolutePath());
            result = gm.increase(incBase, numBatches, comOutBase);
        } else { // batch
            BatchInc bi = new BatchInc();
            bi.initialize(initialPath, initialComPath);
            // BatchInc has no top-level modularity(); emit NaN for batch 0.
            appendMetric(metrics, 0, 0, Double.NaN, -1);
            copyFile(initialComPath, new File(outDir, "com_0.txt").getAbsolutePath());
            result = bi.increase(incBase, numBatches, comOutBase);
        }
        writeMetricsFromResult(metrics, result);
        metrics.close();
    }

    @SuppressWarnings({"unchecked","rawtypes"})
    static void writeMetricsFromResult(PrintWriter metrics, HashMap result) {
        ArrayList modList  = (ArrayList) result.get("modList");
        ArrayList timeList = (ArrayList) result.get("timeList");
        ArrayList comList  = (ArrayList) result.getOrDefault("comList", null);
        int K = modList.size();
        for (int i = 0; i < K; i++) {
            double mod  = ((Number) modList.get(i)).doubleValue();
            double tsec = ((Number) timeList.get(i)).doubleValue(); // seconds
            long ms = (long)(tsec * 1000.0);
            int nc = (comList != null && i < comList.size()) ? ((Number) comList.get(i)).intValue() : -1;
            appendMetric(metrics, i + 1, ms, mod, nc);
        }
    }

    /**
     * DynaMo: stage an initial Network at ntwk2/1, write per-batch inct files, then call
     * ModularityOptimizer_DynaMo.runDynamicModularity. Convert outputs to our schema.
     */
    static void runDynaMo(String initialPath, String incBase, int numBatches, String outDir) throws Exception {
        new File(outDir).mkdirs();
        String tag = "ds"; // dataset directory name inside the staging work dir
        File workDir = new File(outDir, "work");
        File ntwk2 = new File(workDir, "data/" + tag + "/ntwk2");
        File inct  = new File(workDir, "data/" + tag + "/inct");
        ntwk2.mkdirs();
        inct.mkdirs();

        // 1) Build initial Network and serialize. Maintain cumulative edge buffers so we can
        //    cheaply rebuild snapshots ntwk2/(i+1) for each batch.
        ArrayList<int[]>   edgesUV = new ArrayList<>();   // {u,v}
        ArrayList<Double>  edgesW  = new ArrayList<>();
        int maxNode = -1;
        try (BufferedReader r = new BufferedReader(new FileReader(initialPath))) {
            String line;
            while ((line = r.readLine()) != null) {
                String[] t = line.split("\t");
                if (t.length < 2) continue;
                int u = Integer.parseInt(t[0]); int v = Integer.parseInt(t[1]);
                double wt = (t.length > 2) ? Double.parseDouble(t[2]) : 1.0;
                if (u == v) continue;
                if (u > v) { int z = u; u = v; v = z; }
                edgesUV.add(new int[]{u, v});
                edgesW.add(wt);
                if (v > maxNode) maxNode = v;
            }
        }
        Network net = buildNetwork(maxNode + 1, edgesUV, edgesW);
        net.save(new File(ntwk2, "1").getAbsolutePath());
        net = null;  // free for GC

        // 2) For each batch k=1..K, write inct/(k+1) AND build ntwk2/(k+1).
        for (int k = 1; k <= numBatches; k++) {
            String inc = extendFileName(incBase, "_" + k);
            try (BufferedReader r = new BufferedReader(new FileReader(inc));
                 PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(inct, Integer.toString(k+1)))))) {
                String line; int idx = 0;
                while ((line = r.readLine()) != null) {
                    String[] t = line.split("\t");
                    if (t.length < 2) continue;
                    int u = Integer.parseInt(t[0]); int v = Integer.parseInt(t[1]);
                    if (u == v) continue;
                    if (u > v) { int z = u; u = v; v = z; }
                    pw.println(idx + "\t+\t" + u + "\t" + v);
                    idx++;
                    edgesUV.add(new int[]{u, v});
                    edgesW.add(1.0);
                    if (v > maxNode) maxNode = v;
                }
            }
            // Build and save snapshot for batch k+1
            Network snap = buildNetwork(maxNode + 1, edgesUV, edgesW);
            snap.save(new File(ntwk2, Integer.toString(k+1)).getAbsolutePath());
            snap = null;
        }

        // 3) Invoke runDynamicModularity. It uses CWD-relative paths "data/<tag>/...", so chdir.
        // Capture stdout to a buffer.
        String origUserDir = System.getProperty("user.dir");
        try {
            // Java's chdir doesn't change actual process cwd reliably for FileReader, but ModularityOptimizer
            // uses relative paths via new File(...). new File resolves against user.dir? Actually no —
            // new File(relativePath) is resolved against the *current working directory* of the process at
            // the OS level, not user.dir. So we must launch a child process if we wanted a true chdir.
            // Instead: temporarily set user.dir AND copy the workDir contents to current cwd? Cleanest:
            // copy is wasteful. Simpler: do *not* chdir — instead create symlinks from cwd-relative
            // "data/<tag>" pointing into workDir, run, then clean up.
            File cwdData = new File("data");
            cwdData.mkdirs();
            File link = new File(cwdData, tag);
            if (link.exists()) {
                // remove a previous link or directory
                if (java.nio.file.Files.isSymbolicLink(link.toPath())) link.delete();
                else deleteRecursive(link);
            }
            java.nio.file.Files.createSymbolicLink(link.toPath(),
                new File(workDir, "data/" + tag).getAbsoluteFile().toPath());

            // capture stdout
            PrintStream origOut = System.out;
            File stdoutCap = new File(outDir, "dynamo_stdout.log");
            try (PrintStream cap = new PrintStream(new BufferedOutputStream(new FileOutputStream(stdoutCap)))) {
                System.setOut(cap);
                try {
                    ModularityOptimizer_DynaMo.runDynamicModularity(tag, numBatches + 1);
                } finally {
                    System.setOut(origOut);
                }
            }

            // 4) Parse stdout lines that match "<tag>\tDynaMo\t<ibatch>\t<modularity>\t<sumMs>\t<diffMs>\t<otherMs>"
            // and the initial-Louvain pass which uses prefix "<tag>\trunDynamicModularity\t1\t<mod>..."
            // We'll just parse the file <tag>_DynaMo_Modularity_Time which has <mod>\t<time_ms> per batch.
            PrintWriter metrics = new PrintWriter(new BufferedWriter(new FileWriter(new File(outDir, "metrics.tsv"))));
            metrics.println("batch\tlatency_ms\tmodularity\tnum_communities");

            // batch 0: initial Louvain. Parse the FIRST "runDynamicModularity" line from stdout.
            double initMod = Double.NaN; long initMs = 0;
            try (BufferedReader r = new BufferedReader(new FileReader(stdoutCap))) {
                String line;
                while ((line = r.readLine()) != null) {
                    if (line.startsWith(tag + "\trunDynamicModularity\t1\t")) {
                        // tag\trunDynamicModularity\t1\t<mod>\t<time_ms>\t<...>
                        String[] p = line.split("\t");
                        if (p.length >= 5) {
                            try { initMod = Double.parseDouble(p[3]); } catch (Exception e) {}
                            try { initMs  = Long.parseLong(p[4]);     } catch (Exception e) {}
                        }
                        break;
                    }
                }
            }
            // num_communities at batch 0: parse com_1 file
            File com1 = new File(workDir, "data/" + tag + "/runDynamicModularity_" + tag + "_com_1");
            int nc0 = -1;
            if (com1.exists()) {
                java.util.Set<String> cs = new java.util.HashSet<>();
                try (BufferedReader r = new BufferedReader(new FileReader(com1))) {
                    String line; while ((line = r.readLine()) != null) cs.add(line.trim());
                }
                nc0 = cs.size();
                // also write com_0.txt in our flat format (grouped)
                convertNodeClusterFileToFlat(com1, new File(outDir, "com_0.txt"));
            }
            appendMetric(metrics, 0, initMs, initMod, nc0);

            // batches 1..K: read per-batch file <tag>_DynaMo_Modularity_Time
            File mtFile = new File("./" + tag + "_DynaMo_Modularity_Time");
            if (!mtFile.exists()) mtFile = new File(tag + "_DynaMo_Modularity_Time");
            if (mtFile.exists()) {
                try (BufferedReader r = new BufferedReader(new FileReader(mtFile))) {
                    String line; int k = 1;
                    while ((line = r.readLine()) != null && k <= numBatches) {
                        String[] p = line.split("\t");
                        if (p.length < 2) { k++; continue; }
                        double mod = Double.NaN; long ms = 0;
                        try { mod = Double.parseDouble(p[0]); } catch (Exception e) {}
                        try { ms  = Long.parseLong(p[1]);     } catch (Exception e) {}
                        // com file for this batch
                        File comK = new File(workDir, "data/" + tag + "/runDynamicModularity_" + tag + "_com_" + (k+1));
                        int nc = -1;
                        if (comK.exists()) {
                            java.util.Set<String> cs = new java.util.HashSet<>();
                            try (BufferedReader rc = new BufferedReader(new FileReader(comK))) {
                                String s; while ((s = rc.readLine()) != null) cs.add(s.trim());
                            }
                            nc = cs.size();
                            convertNodeClusterFileToFlat(comK, new File(outDir, "com_" + k + ".txt"));
                        }
                        appendMetric(metrics, k, ms, mod, nc);
                        k++;
                    }
                }
            }
            metrics.close();

            // cleanup symlink
            if (java.nio.file.Files.isSymbolicLink(link.toPath())) link.delete();
        } finally {
            // remove the per-tag _DynaMo_Modularity_Time file from cwd (so concurrent runs don't collide)
            File mt = new File(tag + "_DynaMo_Modularity_Time");
            if (mt.exists()) mt.delete();
        }
    }

    /** Build a DynaMo Network from a pre-collected edge buffer with u<v. */
    static Network buildNetwork(int n, ArrayList<int[]> uv, ArrayList<Double> w) {
        int[] nNeighbors = new int[n];
        for (int[] e : uv) { nNeighbors[e[0]]++; nNeighbors[e[1]]++; }
        int[] firstNeighborIndex = new int[n + 1];
        int m = 0;
        for (int i = 0; i < n; i++) { firstNeighborIndex[i] = m; m += nNeighbors[i]; }
        firstNeighborIndex[n] = m;
        int[] neighbor = new int[m];
        double[] edgeWeight = new double[m];
        int[] cnt = new int[n];
        for (int i = 0; i < uv.size(); i++) {
            int[] e = uv.get(i); double wt = w.get(i);
            int u = e[0], v = e[1];
            int j = firstNeighborIndex[u] + cnt[u]++;
            neighbor[j] = v; edgeWeight[j] = wt;
            j = firstNeighborIndex[v] + cnt[v]++;
            neighbor[j] = u; edgeWeight[j] = wt;
        }
        return new Network(n, firstNeighborIndex, neighbor, edgeWeight);
    }

    /** Read an edge list u<TAB>v[<TAB>w] and build a DynaMo Network (undirected CSR). */
    static Network buildNetworkFromEdgeList(String path) throws IOException {
        ArrayList<Integer> a = new ArrayList<>();
        ArrayList<Integer> b = new ArrayList<>();
        ArrayList<Double>  w = new ArrayList<>();
        int maxNode = -1;
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = r.readLine()) != null) {
                String[] t = line.split("\t");
                if (t.length < 2) continue;
                int u = Integer.parseInt(t[0]);
                int v = Integer.parseInt(t[1]);
                double wt = (t.length > 2) ? Double.parseDouble(t[2]) : 1.0;
                if (u == v) continue;
                if (u > v) { int z = u; u = v; v = z; }
                a.add(u); b.add(v); w.add(wt);
                if (v > maxNode) maxNode = v;
            }
        }
        int n = maxNode + 1;
        int[] nNeighbors = new int[n];
        for (int i = 0; i < a.size(); i++) { nNeighbors[a.get(i)]++; nNeighbors[b.get(i)]++; }
        int[] firstNeighborIndex = new int[n + 1];
        int m = 0;
        for (int i = 0; i < n; i++) { firstNeighborIndex[i] = m; m += nNeighbors[i]; }
        firstNeighborIndex[n] = m;
        int[] neighbor = new int[m];
        double[] edgeWeight = new double[m];
        int[] cnt = new int[n];
        for (int i = 0; i < a.size(); i++) {
            int u = a.get(i), v = b.get(i); double wt = w.get(i);
            int j = firstNeighborIndex[u] + cnt[u]++;
            neighbor[j] = v; edgeWeight[j] = wt;
            j = firstNeighborIndex[v] + cnt[v]++;
            neighbor[j] = u; edgeWeight[j] = wt;
        }
        return new Network(n, firstNeighborIndex, neighbor, edgeWeight);
    }

    /** Read a file where line N is the cluster id of node N. Group by cluster, write one
     *  community per line tab-separated members. */
    static void convertNodeClusterFileToFlat(File src, File dst) throws IOException {
        java.util.LinkedHashMap<Integer, ArrayList<Integer>> bag = new java.util.LinkedHashMap<>();
        try (BufferedReader r = new BufferedReader(new FileReader(src))) {
            String line; int idx = 0;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) { idx++; continue; }
                int c;
                try { c = Integer.parseInt(line); } catch (Exception e) { idx++; continue; }
                bag.computeIfAbsent(c, k -> new ArrayList<>()).add(idx);
                idx++;
            }
        }
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(dst)))) {
            for (ArrayList<Integer> members : bag.values()) {
                for (int i = 0; i < members.size(); i++) {
                    if (i > 0) pw.print('\t');
                    pw.print(members.get(i));
                }
                pw.println();
            }
        }
    }

    static void deleteRecursive(File f) {
        if (f.isDirectory()) {
            File[] ch = f.listFiles();
            if (ch != null) for (File c : ch) deleteRecursive(c);
        }
        f.delete();
    }

    static void usage() {
        System.err.println("Usage: BaselineDriver louvain-static <initial> <incBase> <numBatches> <outDir>");
        System.err.println("       BaselineDriver {qca|gremod|batch} <initial> <initialCom> <incBase> <numBatches> <outDir>");
        System.err.println("       BaselineDriver dynamo <initial> <incBase> <numBatches> <outDir>");
        System.exit(2);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) usage();
        switch (args[0]) {
            case "louvain-static":
                if (args.length < 5) usage();
                runLouvainStatic(args[1], args[2], Integer.parseInt(args[3]), args[4]);
                break;
            case "qca": case "gremod": case "batch":
                if (args.length < 6) usage();
                runIncremental(args[0], args[1], args[2], args[3], Integer.parseInt(args[4]), args[5]);
                break;
            case "dynamo":
                if (args.length < 5) usage();
                runDynaMo(args[1], args[2], Integer.parseInt(args[3]), args[4]);
                break;
            default:
                usage();
        }
    }
}
