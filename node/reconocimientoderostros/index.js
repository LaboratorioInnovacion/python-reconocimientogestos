const chokidar = require("chokidar")
const fs = require("fs")
const path = require("path")

const { parseMetaFile } = require("./metaParser")
const { processDetection, getStats } = require("./faceTracker")

const WATCH_PATH =
  "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

const OUTPUT_JSON = "people.json"

console.log("Observando:", WATCH_PATH)

function saveJSON() {
  fs.writeFileSync(OUTPUT_JSON, JSON.stringify(getStats().tracks, null, 2))
}

function waitForJpg(metaPath, retries = 20) {
  return new Promise(resolve => {
    const dir = path.dirname(metaPath)
    const base = path.basename(metaPath, ".meta")
    const jpgPath = path.join(dir, base + ".jpg")

    const check = () => {
      if (fs.existsSync(jpgPath)) return resolve(jpgPath)
      if (retries-- <= 0) return resolve(null)
      setTimeout(check, 300)
    }

    check()
  })
}

const watcher = chokidar.watch(WATCH_PATH, {
  persistent: true,
  ignoreInitial: true,
  depth: 5,
  awaitWriteFinish: {
    stabilityThreshold: 1000,
    pollInterval: 200
  }
})

watcher.on("add", async filePath => {
  if (!filePath.endsWith(".meta")) return

  const detection = parseMetaFile(filePath)
  if (!detection) return

  const jpgPath = await waitForJpg(filePath)
  if (jpgPath) {
    detection.image = jpgPath
  }

  const track = processDetection(detection)
  const stats = getStats()

  console.log(
    `Persona ${track.personId} | Fotos: ${track.snapshots.length} | Activos: ${stats.active} | Únicas: ${stats.totalUnique}`
  )

  saveJSON()
})

// const chokidar = require("chokidar")
// const fs = require("fs")
// const path = require("path")

// const { parseMetaFile } = require("./metaParser")
// const { processDetection, getTracks, getStats } = require("./faceTracker")

// const WATCH_PATH =
//   "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

// const OUTPUT_JSON = "people.json"

// console.log("Observando:", WATCH_PATH)

// function saveJSON() {
//   fs.writeFileSync(OUTPUT_JSON, JSON.stringify(getTracks(), null, 2))
// }

// // ⏳ esperar a que el jpg exista EN LA MISMA CARPETA
// function waitForJpg(metaPath, retries = 15) {
//   return new Promise(resolve => {
//     const dir = path.dirname(metaPath)
//     const base = path.basename(metaPath).replace(".meta", "")
//     const jpgPath = path.join(dir, base + ".jpg")

//     const check = () => {
//       if (fs.existsSync(jpgPath)) return resolve(jpgPath)
//       if (retries-- <= 0) return resolve(null)
//       setTimeout(check, 300)
//     }

//     check()
//   })
// }

// const watcher = chokidar.watch(WATCH_PATH, {
//   persistent: true,
//   ignoreInitial: true,
//   awaitWriteFinish: true,
//   depth: 3 // 🔴 CLAVE: entra en las carpetas GUID
// })

// watcher.on("add", async filePath => {
//   if (!filePath.endsWith(".meta")) return

//   const detection = parseMetaFile(filePath)
//   if (!detection) return

//   const jpgPath = await waitForJpg(filePath)
//   if (jpgPath) detection.image = jpgPath

//   const track = processDetection(detection)
//   const stats = getStats()

//   console.log(
//     `Persona ${track.personId} | Fotos: ${track.snapshots.length} | Activos: ${stats.active} | Únicas: ${stats.totalUnique}`
//   )

//   saveJSON()
// })

// // const chokidar = require("chokidar")
// // const fs = require("fs")
// // const path = require("path")

// // const { parseMetaFile } = require("./metaParser")
// // const { processDetection, getTracks, getStats } = require("./faceTracker")

// // // 🔴 CAMBIÁ SI TU RUTA ES OTRA
// // const WATCH_PATH =
// //   "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

// // const OUTPUT_JSON = "people.json"

// // console.log("Observando:", WATCH_PATH)

// // // ⏳ Esperar JPG (SOLUCIÓN CLAVE)
// // function waitForJpg(metaPath, retries = 15) {
// //   return new Promise(resolve => {
// //     const jpgPath = metaPath.replace(".meta", ".jpg")

// //     const check = () => {
// //       if (fs.existsSync(jpgPath)) {
// //         return resolve(jpgPath)
// //       }
// //       if (retries-- <= 0) {
// //         return resolve(null)
// //       }
// //       setTimeout(check, 300)
// //     }

// //     check()
// //   })
// // }

// // function saveJSON() {
// //   fs.writeFileSync(OUTPUT_JSON, JSON.stringify(getTracks(), null, 2))
// // }

// // const watcher = chokidar.watch(WATCH_PATH, {
// //   ignored: /\.jpg$/,
// //   persistent: true,
// //   ignoreInitial: true,
// //   awaitWriteFinish: true
// // })

// // watcher.on("add", async filePath => {
// //   if (!filePath.endsWith(".meta")) return

// //   const detection = parseMetaFile(filePath)
// //   if (!detection) return

// //   const jpgPath = await waitForJpg(filePath)
// //   if (jpgPath) {
// //     detection.image = jpgPath
// //   }

// //   const track = processDetection(detection)
// //   const stats = getStats()

// //   console.log(
// //     `Persona ${track.personId} | Fotos: ${track.snapshots.length} | Activos: ${stats.activePeople} | Únicas: ${stats.totalUniquePeople}`
// //   )

// //   saveJSON()
// // })

// // // const chokidar = require("chokidar")
// // // const fs = require("fs")
// // // const path = require("path")

// // // const { parseMetaFile } = require("./metaParser")
// // // const { processDetection, getTracks } = require("./faceTracker")

// // // const WATCH_PATH =
// // //   "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

// // // const OUTPUT_JSON = "people.json"

// // // console.log("Observando:", WATCH_PATH)

// // // function saveJSON() {
// // //   fs.writeFileSync(OUTPUT_JSON, JSON.stringify(getTracks(), null, 2))
// // // }

// // // const watcher = chokidar.watch(WATCH_PATH, {
// // //   ignored: /\.jpg$/,
// // //   persistent: true,
// // //   ignoreInitial: true,
// // //   awaitWriteFinish: true
// // // })

// // // watcher.on("add", filePath => {
// // //   if (!filePath.endsWith(".meta")) return

// // //   const detection = parseMetaFile(filePath)
// // //   if (!detection) return

// // //   // 📸 asociar imagen
// // //   const jpgPath = filePath.replace(".meta", ".jpg")
// // //   if (fs.existsSync(jpgPath)) {
// // //     detection.image = jpgPath
// // //   }

// // //   const track = processDetection(detection)

// // //   console.log(
// // //     "Persona:",
// // //     track.personId,
// // //     "| Fotos:",
// // //     track.snapshots.length
// // //   )

// // //   saveJSON()
// // // })

// // // // const chokidar = require("chokidar")
// // // // const path = require("path")

// // // // const { parseMetaFile } = require("./metaParser")
// // // // const { processDetection, getCount } = require("./faceTracker")

// // // // // 🔴 CAMBIÁ ESTA RUTA
// // // // const WATCH_PATH =
// // // //   "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

// // // // console.log("Observando:", WATCH_PATH)

// // // // const watcher = chokidar.watch(WATCH_PATH, {
// // // //   ignored: /\.jpg$/,
// // // //   persistent: true,
// // // //   ignoreInitial: true,
// // // //   awaitWriteFinish: true
// // // // })

// // // // watcher.on("add", filePath => {
// // // //   if (!filePath.endsWith(".meta")) return

// // // //   const detection = parseMetaFile(filePath)
// // // //   if (!detection) return

// // // //   const track = processDetection(detection)

// // // //   console.log(
// // // //     "Persona:",
// // // //     track.id,
// // // //     " | Total:",
// // // //     getCount(),
// // // //     " | Archivo:",
// // // //     path.basename(filePath)
// // // //   )
// // // // })

// // // // // const chokidar = require("chokidar")
// // // // // const path = require("path")

// // // // // const { parseMetaFile } = require("./metaParser")
// // // // // const { processDetection, getCount } = require("./faceTracker")

// // // // // // 🔴 CAMBIÁ ESTA RUTA
// // // // // const WATCH_PATH =
// // // // //   "C:/ProgramData/Camlytics_v3/Data/StorageData/Snapshot"

// // // // // console.log("Observando:", WATCH_PATH)

// // // // // const watcher = chokidar.watch(WATCH_PATH, {
// // // // //   ignored: /\.jpg$/,
// // // // //   persistent: true,
// // // // //   ignoreInitial: true,
// // // // //   awaitWriteFinish: true
// // // // // })

// // // // // watcher.on("add", filePath => {
// // // // //   if (!filePath.endsWith(".meta")) return

// // // // //   const detection = parseMetaFile(filePath)
// // // // //   if (!detection) return

// // // // //   const track = processDetection(detection)

// // // // //   console.log(
// // // // //     "Persona:",
// // // // //     track.id,
// // // // //     " | Total:",
// // // // //     getCount(),
// // // // //     " | Archivo:",
// // // // //     path.basename(filePath)
// // // // //   )
// // // // // })
