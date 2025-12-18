const fs = require("fs")
const { XMLParser } = require("fast-xml-parser")

const parser = new XMLParser({ ignoreAttributes: false })

function parseMetaFile(path) {
  const xml = fs.readFileSync(path, "utf8")
  const json = parser.parse(xml)

  const obj = json?.EventModel?.Object
  if (!obj?.Rect) return null

  return {
    box: {
      left: parseFloat(obj.Rect.Left),
      top: parseFloat(obj.Rect.Top),
      right: parseFloat(obj.Rect.Right),
      bottom: parseFloat(obj.Rect.Bottom)
    },
    time: Date.now() / 1000
  }
}

module.exports = { parseMetaFile }

// const fs = require("fs")
// const { XMLParser } = require("fast-xml-parser")

// const parser = new XMLParser({ ignoreAttributes: false })

// function parseMetaFile(path) {
//   const xml = fs.readFileSync(path, "utf8")
//   const json = parser.parse(xml)

//   const obj = json?.EventModel?.Object
//   if (!obj?.Rect) return null

//   return {
//     box: {
//       left: parseFloat(obj.Rect.Left),
//       top: parseFloat(obj.Rect.Top),
//       right: parseFloat(obj.Rect.Right),
//       bottom: parseFloat(obj.Rect.Bottom)
//     },
//     time: Date.now() / 1000
//   }
// }

// module.exports = { parseMetaFile }

// // const fs = require("fs")
// // const { XMLParser } = require("fast-xml-parser")

// // const parser = new XMLParser({ ignoreAttributes: false })

// // function parseMetaFile(path) {
// //   const xml = fs.readFileSync(path, "utf8")
// //   const json = parser.parse(xml)

// //   const obj = json?.EventModel?.Object
// //   if (!obj?.Rect) return null

// //   return {
// //     box: {
// //       left: parseFloat(obj.Rect.Left),
// //       top: parseFloat(obj.Rect.Top),
// //       right: parseFloat(obj.Rect.Right),
// //       bottom: parseFloat(obj.Rect.Bottom)
// //     },
// //     time: Date.now() / 1000
// //   }
// // }

// // module.exports = { parseMetaFile }

// // // const fs = require("fs")
// // // const { XMLParser } = require("fast-xml-parser")

// // // const parser = new XMLParser({ ignoreAttributes: false })

// // // function parseMetaFile(path) {
// // //   const xml = fs.readFileSync(path, "utf8")
// // //   const json = parser.parse(xml)

// // //   const obj = json?.EventModel?.Object
// // //   if (!obj?.Rect) return null

// // //   return {
// // //     box: {
// // //       left: parseFloat(obj.Rect.Left),
// // //       top: parseFloat(obj.Rect.Top),
// // //       right: parseFloat(obj.Rect.Right),
// // //       bottom: parseFloat(obj.Rect.Bottom)
// // //     },
// // //     time: Date.now() / 1000
// // //   }
// // // }

// // // module.exports = { parseMetaFile }

// // // // const fs = require("fs")
// // // // const { XMLParser } = require("fast-xml-parser")

// // // // const parser = new XMLParser({ ignoreAttributes: false })

// // // // function parseMetaFile(path) {
// // // //   const xml = fs.readFileSync(path, "utf8")
// // // //   const json = parser.parse(xml)

// // // //   const obj = json?.EventModel?.Object
// // // //   if (!obj?.Rect) return null

// // // //   return {
// // // //     box: {
// // // //       left: parseFloat(obj.Rect.Left),
// // // //       top: parseFloat(obj.Rect.Top),
// // // //       right: parseFloat(obj.Rect.Right),
// // // //       bottom: parseFloat(obj.Rect.Bottom)
// // // //     },
// // // //     time: Date.now() / 1000
// // // //   }
// // // // }

// // // // module.exports = { parseMetaFile }
