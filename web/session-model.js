// Supply the existing composer menu with its host catalog before a Session is available.
function studioModelSnapshot(snapshot, selection) {
  return {
    current:selection ?? snapshot.value?.default ?? null,
    groups:snapshot.value?.groups ?? [],
    failures:snapshot.value?.failures ?? [],
    status:snapshot.status, error:snapshot.error, pending:null
  }
}
function studioDraftActions(models, ctx) {
  return {
    available:true, isDraft:true, directory:models.catalog.store,
    load:() => models.catalog.load().catch(() => {}),
    resolveText:text => ctx.locale.resolveText(text),
    select:async selection => {
      models.studioDraft = selection
      return {ok:true}
    }
  }
}
