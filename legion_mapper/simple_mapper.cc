#include "simple_mapper.h"

#include "default_mapper.h"

#ifdef REALM_USE_SUBPROCESSES
// FIXME: Temporary hack until Legion understands allocators.
#include "realm/custom_malloc.h"
#endif

using namespace Legion;
using namespace Legion::Mapping;

class SimpleMapper : public DefaultMapper
{
public:
  SimpleMapper(MapperRuntime *rt, Machine machine, Processor local,
              const char *mapper_name);
  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);
  std::vector<Processor> one_per_remote_pys;
};

SimpleMapper::SimpleMapper(MapperRuntime *rt, Machine machine, Processor local,
                         const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
{
  std::set<AddressSpace> address_spaces;
  for (auto proc : remote_pys) {
    if (!address_spaces.count(proc.address_space())) {
      one_per_remote_pys.push_back(proc);
      address_spaces.insert(proc.address_space());
    }
  }
}

Processor SimpleMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  // If this is an individual task with a point assigned, round robin
  // around the machine.
  if (!task.is_index_space && !task.index_point.is_null()) {
    VariantInfo info = 
      default_find_preferred_variant(task, ctx, false/*needs tight*/);
    switch (info.proc_kind)
    {
      case Processor::LOC_PROC:
        return default_get_next_global_cpu();
      case Processor::TOC_PROC:
        return default_get_next_global_gpu();
      case Processor::IO_PROC:
        return default_get_next_global_io();
      case Processor::OMP_PROC:
        return default_get_next_global_omp();
      case Processor::PY_PROC:
        {
          assert(task.index_point.get_dim() <= 1);
          long long point = task.index_point[0];
          return one_per_remote_pys[(point + 1) % one_per_remote_pys.size()];
        }
      default:
        assert(false);
    }
  }

  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

static void create_mappers(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    SimpleMapper* mapper = new SimpleMapper(runtime->get_mapper_runtime(),
                                          machine, *it, "simple_mapper");
    runtime->replace_default_mapper(mapper, *it);
  }
}

void preregister_simple_mapper()
{
  Runtime::add_registration_callback(create_mappers);
}

void register_simple_mapper()
{
#ifdef REALM_USE_SUBPROCESSES
  INSTALL_REALM_ALLOCATOR;
#endif

  Runtime *runtime = Runtime::get_runtime();
  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery query(machine);
  query.local_address_space();
  std::set<Processor> local_procs(query.begin(), query.end());
  for(auto it = local_procs.begin(); it != local_procs.end(); ) {
    if (it->kind() == Processor::UTIL_PROC) {
      it = local_procs.erase(it);
    } else {
      ++it;
    }
  }
  create_mappers(machine, runtime, local_procs);
}
