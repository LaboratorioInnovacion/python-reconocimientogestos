const { XMLParser } = require("fast-xml-parser")

const parser = new XMLParser({ ignoreAttributes: false })

function parseCamlyticsXML(xml) {
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

module.exports = { parseCamlyticsXML }

// const { XMLParser } = require("fast-xml-parser")

// const parser = new XMLParser({
//   ignoreAttributes: false
// })

// function parseCamlyticsXML(xml) {
//   const json = parser.parse(xml)
//   const obj = json.EventModel?.Object

//   if (!obj) return null

//   return {
//     x: parseFloat(obj.Position.X),
//     y: parseFloat(obj.Position.Y),
//     age: parseInt(obj.Age),
//     gender: obj.Gender,
//     time: Date.now() / 1000
//   }
// }

// module.exports = { parseCamlyticsXML }
