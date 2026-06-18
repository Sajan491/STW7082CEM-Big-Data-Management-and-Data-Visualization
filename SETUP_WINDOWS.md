# Windows Setup Guide - PySpark Environment

**Project:** Kickstarter Campaign Analytics (STW7082CEM)
**Student ID:** 250289
**Purpose:** Reproducible local setup so `Kickstarter_Analysis.ipynb` runs end-to-end on Windows.

PySpark runs on the Java JVM and uses Hadoop's filesystem code. On Windows three things must
be in place that `pip install pyspark` does **not** provide on its own:

1. **A Java JDK** (the JVM that Spark runs on)
2. **The Python packages** (`pyspark`, `pandas`, etc.)
3. **Hadoop native binaries** (`winutils.exe` + `hadoop.dll`) - Windows-only requirement

Complete all three steps below, then restart your terminal / VS Code so the environment
variables load.

---

## Step 1 - Install a Java JDK

Spark 3.5.x works with Java 8, 11, or 17 **only**. Newer Java (e.g. 21/25) breaks Spark 3.5
with reflective-access errors, so use **Java 17 (Eclipse Temurin)**.

> This project's JDK 17 is installed at:
> `C:\Users\sajan.mahat\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot`
> (JDK 25 is also installed but is *not* used - `JAVA_HOME` points at 17 below.)

Download Temurin 17 if needed: https://adoptium.net

Run this block (sets `JAVA_HOME` to 17, puts its `bin` first on PATH so it wins over JDK 25,
and applies to the current session):

```powershell
$jdk17 = "C:\Users\sajan.mahat\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
[Environment]::SetEnvironmentVariable("JAVA_HOME", $jdk17, "User")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$jdk17\bin;$userPath", "User")
$env:JAVA_HOME = $jdk17
$env:Path = "$jdk17\bin;$env:Path"
java -version
```

`java -version` should print `openjdk version "17.0.19"`. After this, **restart VS Code** so the
Jupyter kernel inherits Java 17.

---

## Step 2 - Install Python Packages

From the project folder, install the pinned dependencies:

```powershell
pip install -r requirements.txt
```

This installs `pyspark==3.5.4`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and the
Jupyter/ipykernel runtime. Verify:

```powershell
python -c "import pyspark; print(pyspark.__version__)"
```

---

## Step 3 - Hadoop Native Binaries (`winutils.exe` + `hadoop.dll`)

**Why:** Spark uses Hadoop's filesystem layer, which on Windows expects native binaries that
are not bundled with PySpark. Without them you will see errors such as:

- `java.io.IOException: Could not locate executable null\bin\winutils.exe`
- `UnsatisfiedLinkError: ... hadoop.dll`

**Version:** PySpark 3.5.4 bundles **Hadoop 3.3.4**. The closest build in the community repo is
**`hadoop-3.3.5`**, which is binary-compatible and works fine (mismatched major versions can
fail to load).

**Source:** `cdarlint/winutils` GitHub repo - https://github.com/cdarlint/winutils
(the most widely used source in the Spark community; these are third-party executables, so use
only if you are comfortable with that).

### Instructions

1. Create the folder and download both files (Hadoop 3.3.5 build):

   ```powershell
   New-Item -ItemType Directory -Force "C:\hadoop\bin" | Out-Null

   $base = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin"
   Invoke-WebRequest "$base/winutils.exe" -OutFile "C:\hadoop\bin\winutils.exe"
   Invoke-WebRequest "$base/hadoop.dll"   -OutFile "C:\hadoop\bin\hadoop.dll"

   Get-ChildItem "C:\hadoop\bin"   # confirm both files are present
   ```

   If a proxy/AV blocks the download, open the repo in a browser and save the two files from
   `hadoop-3.3.5/bin/` into `C:\hadoop\bin` manually.

2. Set the environment variables (user-level + current session):

   ```powershell
   [Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
   $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
   [Environment]::SetEnvironmentVariable("Path", "$userPath;C:\hadoop\bin", "User")
   $env:HADOOP_HOME = "C:\hadoop"
   $env:Path = "$env:Path;C:\hadoop\bin"
   ```

3. (If `hadoop.dll` still is not found at runtime) copy it into System32:

   ```powershell
   Copy-Item "C:\hadoop\bin\hadoop.dll" "C:\Windows\System32\" -Force
   ```

---

## Step 4 - Verify the Full Setup

Open a **new** terminal (so all variables are loaded) and run:

```powershell
java -version                                  # Step 1 - JDK present
python -c "import pyspark; print(pyspark.__version__)"   # Step 2 - PySpark present
echo $env:HADOOP_HOME                           # Step 3 - should print C:\hadoop
Test-Path "C:\hadoop\bin\winutils.exe"          # should print True
Test-Path "C:\hadoop\bin\hadoop.dll"            # should print True
```

Then open `Kickstarter_Analysis.ipynb` and run the Phase 1 cells. A successful
`SparkSession` start with no `winutils`/`hadoop.dll` warnings confirms the environment is ready.

---

## Quick Reference - Required Environment Variables

| Variable      | Value                                                                          | Set in |
|---------------|--------------------------------------------------------------------------------|--------|
| `JAVA_HOME`   | `C:\Users\sajan.mahat\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.19.10-hotspot` | Step 1 |
| `HADOOP_HOME` | `C:\hadoop`                                                                     | Step 3 |
| `PATH` (adds) | `<JAVA_HOME>\bin` (first) and `C:\hadoop\bin`                                   | Steps 1 & 3 |

---

## Common Errors

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Could not locate executable ...\winutils.exe` | `winutils.exe` missing / `HADOOP_HOME` not set | Step 3 |
| `UnsatisfiedLinkError: ... hadoop.dll` | `hadoop.dll` missing or wrong version | Step 3 (match Hadoop 3.3.x) |
| `JAVA_HOME is not set` / `java not recognized` | JDK not installed or not on PATH | Step 1 |
| `No module named 'pyspark'` | Packages not installed | Step 2 |
| `Python worker failed to connect` | Notebook kernel uses a different Python than PySpark | Cell 1.1 sets `PYSPARK_PYTHON = sys.executable` - re-run it |
