let tracks = []
let nextId = 1
let totalUnique = 0

const IOU_THRESHOLD = 0.3
const DIST_THRESHOLD = 0.05
const DISAPPEAR_TIME = 60

function center(b) {
  return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function iou(a, b) {
  const xA = Math.max(a.left, b.left)
  const yA = Math.max(a.top, b.top)
  const xB = Math.min(a.right, b.right)
  const yB = Math.min(a.bottom, b.bottom)

  const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
  const areaA = (a.right - a.left) * (a.bottom - a.top)
  const areaB = (b.right - b.left) * (b.bottom - b.top)

  return inter / (areaA + areaB - inter || 1)
}

function cleanup(now) {
  tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
}

function processDetection(det) {
  cleanup(det.time)

  for (const t of tracks) {
    if (
      iou(t.box, det.box) > IOU_THRESHOLD ||
      distance(center(t.box), center(det.box)) < DIST_THRESHOLD
    ) {
      t.box = det.box
      t.lastSeen = det.time
      if (det.image && !t.snapshots.includes(det.image)) {
        t.snapshots.push(det.image)
      }
      return t
    }
  }

  totalUnique++

  const newTrack = {
    personId: nextId++,
    box: det.box,
    firstSeen: new Date(det.time * 1000).toISOString(),
    lastSeen: new Date(det.time * 1000).toISOString(),
    snapshots: det.image ? [det.image] : []
  }

  tracks.push(newTrack)
  return newTrack
}

function getStats() {
  return {
    active: tracks.length,
    unique: totalUnique,
    tracks
  }
}

module.exports = { processDetection, getStats }

// let tracks = []
// let nextId = 1
// let totalUniquePeople = 0

// const IOU_THRESHOLD = 0.25
// const DIST_THRESHOLD = 0.06
// const TIME_WINDOW = 15       // Camlytics es lento
// const DISAPPEAR_TIME = 120   // 2 minutos

// function center(b) {
//   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// }

// function distance(a, b) {
//   const dx = a.x - b.x
//   const dy = a.y - b.y
//   return Math.sqrt(dx * dx + dy * dy)
// }

// function iou(a, b) {
//   const xA = Math.max(a.left, b.left)
//   const yA = Math.max(a.top, b.top)
//   const xB = Math.min(a.right, b.right)
//   const yB = Math.min(a.bottom, b.bottom)

//   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
//   const areaA = (a.right - a.left) * (a.bottom - a.top)
//   const areaB = (b.right - b.left) * (b.bottom - b.top)

//   return inter / (areaA + areaB - inter || 1)
// }

// function cleanup(now) {
//   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// }

// function isSameTrack(track, det) {
//   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false

//   if (iou(track.box, det.box) > IOU_THRESHOLD) return true

//   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// }

// function processDetection(det) {
//   cleanup(det.time)

//   for (const t of tracks) {
//     if (isSameTrack(t, det)) {
//       t.box = det.box
//       t.lastSeen = det.time

//       if (det.image && !t.snapshots.includes(det.image)) {
//         t.snapshots.push(det.image)
//       }
//       return t
//     }
//   }

//   totalUniquePeople++

//   const newTrack = {
//     personId: nextId++,
//     box: det.box,
//     firstSeen: new Date(det.time * 1000).toISOString(),
//     lastSeen: new Date(det.time * 1000).toISOString(),
//     snapshots: det.image ? [det.image] : []
//   }

//   tracks.push(newTrack)
//   return newTrack
// }

// function getStats() {
//   return {
//     active: tracks.length,
//     totalUnique: totalUniquePeople
//   }
// }

// function getTracks() {
//   return tracks
// }

// module.exports = { processDetection, getTracks, getStats }

// // let tracks = []
// // let nextId = 1
// // let totalUnique = 0

// // const IOU_THRESHOLD = 0.3
// // const DIST_THRESHOLD = 0.06
// // const TIME_WINDOW = 15       // Camlytics es lento
// // const DISAPPEAR_TIME = 120   // 2 minutos

// // function center(b) {
// //   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// // }

// // function distance(a, b) {
// //   return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
// // }

// // function iou(a, b) {
// //   const xA = Math.max(a.left, b.left)
// //   const yA = Math.max(a.top, b.top)
// //   const xB = Math.min(a.right, b.right)
// //   const yB = Math.min(a.bottom, b.bottom)

// //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// //   return inter / (areaA + areaB - inter || 1)
// // }

// // function cleanup(now) {
// //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // }

// // function isSameTrack(track, det) {
// //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false
// //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true
// //   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// // }

// // function processDetection(det) {
// //   cleanup(det.time)

// //   for (const t of tracks) {
// //     if (isSameTrack(t, det)) {
// //       t.box = det.box
// //       t.lastSeen = det.time
// //       if (det.image && !t.snapshots.includes(det.image)) {
// //         t.snapshots.push(det.image)
// //       }
// //       return t
// //     }
// //   }

// //   totalUnique++

// //   const newTrack = {
// //     personId: nextId++,
// //     box: det.box,
// //     firstSeen: new Date(det.time * 1000).toISOString(),
// //     lastSeen: new Date(det.time * 1000).toISOString(),
// //     snapshots: det.image ? [det.image] : []
// //   }

// //   tracks.push(newTrack)
// //   return newTrack
// // }

// // function getStats() {
// //   return {
// //     active: tracks.length,
// //     unique: totalUnique,
// //     tracks
// //   }
// // }

// // module.exports = { processDetection, getStats }

// // // let tracks = []
// // // let nextId = 1
// // // let totalUnique = 0

// // // const IOU_THRESHOLD = 0.3
// // // const DIST_THRESHOLD = 0.06
// // // const TIME_WINDOW = 15
// // // const DISAPPEAR_TIME = 120

// // // function center(b) {
// // //   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// // // }

// // // function distance(a, b) {
// // //   const dx = a.x - b.x
// // //   const dy = a.y - b.y
// // //   return Math.sqrt(dx * dx + dy * dy)
// // // }

// // // function iou(a, b) {
// // //   const xA = Math.max(a.left, b.left)
// // //   const yA = Math.max(a.top, b.top)
// // //   const xB = Math.min(a.right, b.right)
// // //   const yB = Math.min(a.bottom, b.bottom)

// // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // //   return inter / (areaA + areaB - inter || 1)
// // // }

// // // function cleanup(now) {
// // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // }

// // // function isSameTrack(track, det) {
// // //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false
// // //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true
// // //   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// // // }

// // // function processDetection(det) {
// // //   cleanup(det.time)

// // //   for (const t of tracks) {
// // //     if (isSameTrack(t, det)) {
// // //       t.box = det.box
// // //       t.lastSeen = det.time
// // //       if (det.image && !t.snapshots.includes(det.image)) {
// // //         t.snapshots.push(det.image)
// // //       }
// // //       return t
// // //     }
// // //   }

// // //   totalUnique++

// // //   const newTrack = {
// // //     personId: nextId++,
// // //     box: det.box,
// // //     firstSeen: new Date(det.time * 1000).toISOString(),
// // //     lastSeen: new Date(det.time * 1000).toISOString(),
// // //     snapshots: det.image ? [det.image] : []
// // //   }

// // //   tracks.push(newTrack)
// // //   return newTrack
// // // }

// // // function getStats() {
// // //   return {
// // //     active: tracks.length,
// // //     totalUnique,
// // //     tracks
// // //   }
// // // }

// // // module.exports = { processDetection, getStats }

// // // // let tracks = []
// // // // let nextId = 1
// // // // let totalUniquePeople = 0

// // // // const IOU_THRESHOLD = 0.3
// // // // const DIST_THRESHOLD = 0.05
// // // // const TIME_WINDOW = 10
// // // // const DISAPPEAR_TIME = 120

// // // // function center(b) {
// // // //   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// // // // }

// // // // function distance(a, b) {
// // // //   const dx = a.x - b.x
// // // //   const dy = a.y - b.y
// // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // }

// // // // function iou(a, b) {
// // // //   const xA = Math.max(a.left, b.left)
// // // //   const yA = Math.max(a.top, b.top)
// // // //   const xB = Math.min(a.right, b.right)
// // // //   const yB = Math.min(a.bottom, b.bottom)

// // // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // // //   return inter / (areaA + areaB - inter || 1)
// // // // }

// // // // function cleanup(now) {
// // // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // // }

// // // // function isSameTrack(track, det) {
// // // //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false
// // // //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true
// // // //   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// // // // }

// // // // function processDetection(det) {
// // // //   cleanup(det.time)

// // // //   for (const t of tracks) {
// // // //     if (isSameTrack(t, det)) {
// // // //       t.box = det.box
// // // //       t.lastSeen = det.time

// // // //       if (det.image && !t.snapshots.includes(det.image)) {
// // // //         t.snapshots.push(det.image)
// // // //       }

// // // //       return t
// // // //     }
// // // //   }

// // // //   totalUniquePeople++

// // // //   const newTrack = {
// // // //     personId: nextId++,
// // // //     box: det.box,
// // // //     firstSeen: new Date(det.time * 1000).toISOString(),
// // // //     lastSeen: new Date(det.time * 1000).toISOString(),
// // // //     snapshots: det.image ? [det.image] : []
// // // //   }

// // // //   tracks.push(newTrack)
// // // //   return newTrack
// // // // }

// // // // function getStats() {
// // // //   return {
// // // //     active: tracks.length,
// // // //     totalUnique: totalUniquePeople
// // // //   }
// // // // }

// // // // function getTracks() {
// // // //   return tracks
// // // // }

// // // // module.exports = { processDetection, getTracks, getStats }

// // // // // let tracks = []
// // // // // let nextId = 1
// // // // // let totalUniquePeople = 0

// // // // // // 🔧 AJUSTES IMPORTANTES PARA CAMLYTICS
// // // // // const IOU_THRESHOLD = 0.3
// // // // // const DIST_THRESHOLD = 0.06
// // // // // const TIME_WINDOW = 10        // segundos
// // // // // const DISAPPEAR_TIME = 120    // segundos (CLAVE)

// // // // // function center(b) {
// // // // //   return {
// // // // //     x: (b.left + b.right) / 2,
// // // // //     y: (b.top + b.bottom) / 2
// // // // //   }
// // // // // }

// // // // // function distance(a, b) {
// // // // //   const dx = a.x - b.x
// // // // //   const dy = a.y - b.y
// // // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // // }

// // // // // function iou(a, b) {
// // // // //   const xA = Math.max(a.left, b.left)
// // // // //   const yA = Math.max(a.top, b.top)
// // // // //   const xB = Math.min(a.right, b.right)
// // // // //   const yB = Math.min(a.bottom, b.bottom)

// // // // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // // // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // // // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // // // //   return inter / (areaA + areaB - inter || 1)
// // // // // }

// // // // // function cleanup(now) {
// // // // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // // // }

// // // // // function isSameTrack(track, det) {
// // // // //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false

// // // // //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true

// // // // //   const c1 = center(track.box)
// // // // //   const c2 = center(det.box)
// // // // //   return distance(c1, c2) < DIST_THRESHOLD
// // // // // }

// // // // // function processDetection(det) {
// // // // //   cleanup(det.time)

// // // // //   for (const t of tracks) {
// // // // //     if (isSameTrack(t, det)) {
// // // // //       t.box = det.box
// // // // //       t.lastSeen = det.time

// // // // //       if (det.image && !t.snapshots.includes(det.image)) {
// // // // //         t.snapshots.push(det.image)
// // // // //       }

// // // // //       return t
// // // // //     }
// // // // //   }

// // // // //   // ➕ PERSONA NUEVA REAL
// // // // //   totalUniquePeople++

// // // // //   const newTrack = {
// // // // //     personId: nextId++,
// // // // //     box: det.box,
// // // // //     firstSeen: new Date(det.time * 1000).toISOString(),
// // // // //     lastSeen: new Date(det.time * 1000).toISOString(),
// // // // //     snapshots: det.image ? [det.image] : []
// // // // //   }

// // // // //   tracks.push(newTrack)
// // // // //   return newTrack
// // // // // }

// // // // // function getTracks() {
// // // // //   return tracks
// // // // // }

// // // // // function getStats() {
// // // // //   return {
// // // // //     activePeople: tracks.length,
// // // // //     totalUniquePeople
// // // // //   }
// // // // // }

// // // // // module.exports = {
// // // // //   processDetection,
// // // // //   getTracks,
// // // // //   getStats
// // // // // }

// // // // // // let tracks = []
// // // // // // let nextId = 1

// // // // // // const IOU_THRESHOLD = 0.3
// // // // // // const DIST_THRESHOLD = 0.05
// // // // // // const TIME_WINDOW = 2.5
// // // // // // const DISAPPEAR_TIME = 8

// // // // // // function center(b) {
// // // // // //   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// // // // // // }

// // // // // // function distance(a, b) {
// // // // // //   const dx = a.x - b.x
// // // // // //   const dy = a.y - b.y
// // // // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // // // }

// // // // // // function iou(a, b) {
// // // // // //   const xA = Math.max(a.left, b.left)
// // // // // //   const yA = Math.max(a.top, b.top)
// // // // // //   const xB = Math.min(a.right, b.right)
// // // // // //   const yB = Math.min(a.bottom, b.bottom)

// // // // // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // // // // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // // // // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // // // // //   return inter / (areaA + areaB - inter || 1)
// // // // // // }

// // // // // // function cleanup(now) {
// // // // // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // // // // }

// // // // // // function isSameTrack(track, det) {
// // // // // //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false

// // // // // //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true

// // // // // //   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// // // // // // }

// // // // // // function processDetection(det) {
// // // // // //   cleanup(det.time)

// // // // // //   for (const t of tracks) {
// // // // // //     if (isSameTrack(t, det)) {
// // // // // //       t.box = det.box
// // // // // //       t.lastSeen = det.time

// // // // // //       if (det.image && !t.snapshots.includes(det.image)) {
// // // // // //         t.snapshots.push(det.image)
// // // // // //       }

// // // // // //       return t
// // // // // //     }
// // // // // //   }

// // // // // //   const newTrack = {
// // // // // //     personId: nextId++,
// // // // // //     box: det.box,
// // // // // //     firstSeen: new Date(det.time * 1000).toISOString(),
// // // // // //     lastSeen: new Date(det.time * 1000).toISOString(),
// // // // // //     snapshots: det.image ? [det.image] : []
// // // // // //   }

// // // // // //   tracks.push(newTrack)
// // // // // //   return newTrack
// // // // // // }

// // // // // // function getTracks() {
// // // // // //   return tracks
// // // // // // }

// // // // // // module.exports = { processDetection, getTracks }

// // // // // // // let tracks = []
// // // // // // // let nextId = 1

// // // // // // // const IOU_THRESHOLD = 0.3
// // // // // // // const DIST_THRESHOLD = 0.05
// // // // // // // const TIME_WINDOW = 2.5
// // // // // // // const DISAPPEAR_TIME = 8

// // // // // // // function center(b) {
// // // // // // //   return { x: (b.left + b.right) / 2, y: (b.top + b.bottom) / 2 }
// // // // // // // }

// // // // // // // function distance(a, b) {
// // // // // // //   const dx = a.x - b.x
// // // // // // //   const dy = a.y - b.y
// // // // // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // // // // }

// // // // // // // function iou(a, b) {
// // // // // // //   const xA = Math.max(a.left, b.left)
// // // // // // //   const yA = Math.max(a.top, b.top)
// // // // // // //   const xB = Math.min(a.right, b.right)
// // // // // // //   const yB = Math.min(a.bottom, b.bottom)

// // // // // // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // // // // // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // // // // // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // // // // // //   return inter / (areaA + areaB - inter || 1)
// // // // // // // }

// // // // // // // function cleanup(now) {
// // // // // // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // // // // // }

// // // // // // // function isSameTrack(track, det) {
// // // // // // //   if (Math.abs(track.lastSeen - det.time) > TIME_WINDOW) return false

// // // // // // //   if (iou(track.box, det.box) > IOU_THRESHOLD) return true

// // // // // // //   return distance(center(track.box), center(det.box)) < DIST_THRESHOLD
// // // // // // // }

// // // // // // // function processDetection(det) {
// // // // // // //   cleanup(det.time)

// // // // // // //   for (const t of tracks) {
// // // // // // //     if (isSameTrack(t, det)) {
// // // // // // //       t.box = det.box
// // // // // // //       t.lastSeen = det.time
// // // // // // //       return t
// // // // // // //     }
// // // // // // //   }

// // // // // // //   const newTrack = {
// // // // // // //     id: nextId++,
// // // // // // //     box: det.box,
// // // // // // //     firstSeen: det.time,
// // // // // // //     lastSeen: det.time
// // // // // // //   }

// // // // // // //   tracks.push(newTrack)
// // // // // // //   return newTrack
// // // // // // // }

// // // // // // // function getCount() {
// // // // // // //   return tracks.length
// // // // // // // }

// // // // // // // module.exports = { processDetection, getCount }

// // // // // // // // let tracks = []
// // // // // // // // let nextId = 1

// // // // // // // // const IOU_THRESHOLD = 0.3
// // // // // // // // const DIST_THRESHOLD = 0.05
// // // // // // // // const TIME_WINDOW = 2.5     // segundos
// // // // // // // // const DISAPPEAR_TIME = 8    // segundos

// // // // // // // // function center(box) {
// // // // // // // //   return {
// // // // // // // //     x: (box.left + box.right) / 2,
// // // // // // // //     y: (box.top + box.bottom) / 2
// // // // // // // //   }
// // // // // // // // }

// // // // // // // // function distance(a, b) {
// // // // // // // //   const dx = a.x - b.x
// // // // // // // //   const dy = a.y - b.y
// // // // // // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // // // // // }

// // // // // // // // function iou(a, b) {
// // // // // // // //   const xA = Math.max(a.left, b.left)
// // // // // // // //   const yA = Math.max(a.top, b.top)
// // // // // // // //   const xB = Math.min(a.right, b.right)
// // // // // // // //   const yB = Math.min(a.bottom, b.bottom)

// // // // // // // //   const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA)
// // // // // // // //   const areaA = (a.right - a.left) * (a.bottom - a.top)
// // // // // // // //   const areaB = (b.right - b.left) * (b.bottom - b.top)

// // // // // // // //   return inter / (areaA + areaB - inter || 1)
// // // // // // // // }

// // // // // // // // function cleanup(now) {
// // // // // // // //   tracks = tracks.filter(t => now - t.lastSeen < DISAPPEAR_TIME)
// // // // // // // // }

// // // // // // // // function isSameTrack(track, detection) {
// // // // // // // //   const timeDiff = Math.abs(track.lastSeen - detection.time)
// // // // // // // //   if (timeDiff > TIME_WINDOW) return false

// // // // // // // //   const iouVal = iou(track.box, detection.box)
// // // // // // // //   if (iouVal > IOU_THRESHOLD) return true

// // // // // // // //   const c1 = center(track.box)
// // // // // // // //   const c2 = center(detection.box)

// // // // // // // //   return distance(c1, c2) < DIST_THRESHOLD
// // // // // // // // }

// // // // // // // // function processDetection(detection) {
// // // // // // // //   cleanup(detection.time)

// // // // // // // //   for (const track of tracks) {
// // // // // // // //     if (isSameTrack(track, detection)) {
// // // // // // // //       track.box = detection.box
// // // // // // // //       track.lastSeen = detection.time
// // // // // // // //       return track
// // // // // // // //     }
// // // // // // // //   }

// // // // // // // //   const newTrack = {
// // // // // // // //     id: nextId++,
// // // // // // // //     box: detection.box,
// // // // // // // //     firstSeen: detection.time,
// // // // // // // //     lastSeen: detection.time,
// // // // // // // //     counted: true
// // // // // // // //   }

// // // // // // // //   tracks.push(newTrack)
// // // // // // // //   return newTrack
// // // // // // // // }

// // // // // // // // function getCount() {
// // // // // // // //   return tracks.length
// // // // // // // // }

// // // // // // // // module.exports = {
// // // // // // // //   processDetection,
// // // // // // // //   getCount
// // // // // // // // }

// // // // // // // // // let faces = []
// // // // // // // // // let personIdCounter = 1

// // // // // // // // // const MAX_DISTANCE = 0.04
// // // // // // // // // const MAX_AGE_DIFF = 3
// // // // // // // // // const TIME_WINDOW = 3       // segundos
// // // // // // // // // const DISAPPEAR_TIME = 10   // segundos

// // // // // // // // // function distance(a, b) {
// // // // // // // // //   const dx = a.x - b.x
// // // // // // // // //   const dy = a.y - b.y
// // // // // // // // //   return Math.sqrt(dx * dx + dy * dy)
// // // // // // // // // }

// // // // // // // // // function isSamePerson(a, b) {
// // // // // // // // //   return (
// // // // // // // // //     distance(a, b) < MAX_DISTANCE &&
// // // // // // // // //     Math.abs(a.age - b.age) <= MAX_AGE_DIFF &&
// // // // // // // // //     a.gender === b.gender &&
// // // // // // // // //     Math.abs(a.time - b.time) <= TIME_WINDOW
// // // // // // // // //   )
// // // // // // // // // }

// // // // // // // // // function cleanup(now) {
// // // // // // // // //   faces = faces.filter(f => now - f.lastSeen < DISAPPEAR_TIME)
// // // // // // // // // }

// // // // // // // // // function processFace(face) {
// // // // // // // // //   const now = face.time

// // // // // // // // //   cleanup(now)

// // // // // // // // //   for (const tracked of faces) {
// // // // // // // // //     if (isSamePerson(tracked, face)) {
// // // // // // // // //       tracked.x = face.x
// // // // // // // // //       tracked.y = face.y
// // // // // // // // //       tracked.lastSeen = now
// // // // // // // // //       return tracked
// // // // // // // // //     }
// // // // // // // // //   }

// // // // // // // // //   const newFace = {
// // // // // // // // //     id: personIdCounter++,
// // // // // // // // //     x: face.x,
// // // // // // // // //     y: face.y,
// // // // // // // // //     age: face.age,
// // // // // // // // //     gender: face.gender,
// // // // // // // // //     firstSeen: now,
// // // // // // // // //     lastSeen: now,
// // // // // // // // //     counted: true
// // // // // // // // //   }

// // // // // // // // //   faces.push(newFace)
// // // // // // // // //   return newFace
// // // // // // // // // }

// // // // // // // // // function getCount() {
// // // // // // // // //   return faces.length
// // // // // // // // // }

// // // // // // // // // module.exports = {
// // // // // // // // //   processFace,
// // // // // // // // //   getCount
// // // // // // // // // }
