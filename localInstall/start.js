const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Use the directory where the executable is located (e.g., Desktop/hax/)
const baseDir = path.dirname(process.execPath);
const modulePath = path.join(baseDir, 'node_modules', '@haxtheweb', 'create');
const pkgJsonPath = path.join(modulePath, 'package.json');

// Function to get the binary path from package.json
function getHaxBinPath() {
  if (fs.existsSync(pkgJsonPath)) {
    const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
    if (pkg.bin) {
      const binEntry = typeof pkg.bin === 'string' ? pkg.bin : pkg.bin.hax;
      if (binEntry) {
        return path.join(modulePath, binEntry);
      }
    }
  }
  // Fallback: Guess common locations
  const fallbackPath = path.join(modulePath, 'bin', 'hax.js');
  return fs.existsSync(fallbackPath) ? fallbackPath : null;
}

// Function to install @haxtheweb/create if not present
function ensureModuleInstalled() {
  const haxBinPath = getHaxBinPath();
  if (!haxBinPath || !fs.existsSync(haxBinPath)) {
    console.log('@haxtheweb/create binary not found. Installing it now...');
    try {
      execSync('npm install @haxtheweb/create --no-save --prefix .', {
        stdio: 'inherit',
        cwd: baseDir,
      });
      const newHaxBinPath = getHaxBinPath();
      if (!newHaxBinPath || !fs.existsSync(newHaxBinPath)) {
        throw new Error('Installation succeeded, but hax binary not found');
      }
      console.log('Installation complete. hax binary located at:', newHaxBinPath);
      return newHaxBinPath;
    } catch (error) {
      console.error('Failed to install @haxtheweb/create:', error.message);
      process.exit(1);
    }
  } else {
    console.log('@haxtheweb/create is already installed. hax binary at:', haxBinPath);
    return haxBinPath;
  }
}

// Get command-line arguments
const args = process.argv.slice(2);

// If no arguments (e.g., double-click), just ensure the module is installed and exit
if (args.length === 0) {
  ensureModuleInstalled();
  console.log('Setup complete. Run with a command, e.g., "./hax site" or "./hax start" from this folder.');
  process.exit(0);
}

// Run the specified hax command
const command = args.join(' ');
try {
  const haxBinPath = ensureModuleInstalled();
  console.log('Running command:', `node "${haxBinPath}" ${command}`);
  execSync(`node "${haxBinPath}" ${command}`, { stdio: 'inherit', cwd: process.cwd() });
} catch (error) {
  console.error(`Error running hax ${command}:`, error.message);
  process.exit(1);
}