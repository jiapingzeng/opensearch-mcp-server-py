// Mapping view for IndexMappingTool.
var mappingState = newMappingState();

function renderMapping(data) {
  var m = data.meta || {};
  var names = data.indices.map(function (i) { return i.name; });
  app.appendChild(header(data, 'Mapping: ' + names.join(', '), ['fields: ' + formatNumber(m.field_count || 0)]));
  app.appendChild(mappingPanel(data, mappingState));
}

startView({
  name: 'opensearch-mapping',
  unsupportedText: 'This response has no mapping tree; the text result is unchanged.',
  accepts: function (data) { return data.kind === 'index_mapping' && Array.isArray(data.indices); },
  render: renderMapping
});
